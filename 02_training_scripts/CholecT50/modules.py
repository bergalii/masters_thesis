from collections import OrderedDict
from torchvision.models.video.swin_transformer import (
    swin3d_s,
    Swin3D_S_Weights,
)
from typing import Dict, List, Optional, Callable, Set, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
import types
from torchvision.ops.misc import Conv3dNormActivation
from torchvision.utils import _log_api_usage_once


class CrossTaskAttention(nn.Module):
    """Cross-task attention mechanism for I-V-T components"""

    def __init__(self, common_dim, num_heads=4):
        super().__init__()
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.head_dim = common_dim // num_heads

        # Create task-specific projections
        # Each task has its own Q, K, V projections
        self.task_projections = nn.ModuleDict(
            {
                "instrument": self._create_projections(),
                "verb": self._create_projections(),
                "target": self._create_projections(),
            }
        )

        # Output projections
        self.output_projections = nn.ModuleDict(
            {
                "instrument": nn.Linear(common_dim, common_dim),
                "verb": nn.Linear(common_dim, common_dim),
                "target": nn.Linear(common_dim, common_dim),
            }
        )

        # Layer norms for each task
        self.layer_norms = nn.ModuleDict(
            {
                "instrument": nn.LayerNorm(common_dim),
                "verb": nn.LayerNorm(common_dim),
                "target": nn.LayerNorm(common_dim),
            }
        )

        self.dropout = nn.Dropout(0.1)

    def _create_projections(self):
        """Create query, key, value projections for a task"""
        return nn.ModuleDict(
            {
                "query": nn.Linear(self.common_dim, self.common_dim),
                "key": nn.Linear(self.common_dim, self.common_dim),
                "value": nn.Linear(self.common_dim, self.common_dim),
            }
        )

    def _split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

    def _combine_heads(self, x):
        """Combine the heads back into original shape"""
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        return x.contiguous().view(batch_size, -1, self.common_dim)

    def _scale_dot_product_attention(self, q, k, v):
        """Compute scaled dot-product attention"""
        d_k = torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=q.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output

    def forward(self, inst_emb, verb_emb, target_emb):
        """
        Apply cross-task attention

        Args:
            inst_emb: Instrument embeddings [batch_size, common_dim]
            verb_emb: Verb embeddings [batch_size, common_dim]
            target_emb: Target embeddings [batch_size, common_dim]

        Returns:
            Updated embeddings for each task
        """

        # Store original embeddings for residual connections
        inst_residual = inst_emb
        verb_residual = verb_emb
        target_residual = target_emb

        # Create task-specific queries
        inst_q = self._split_heads(
            self.task_projections["instrument"]["query"](inst_emb).unsqueeze(1)
        )
        verb_q = self._split_heads(
            self.task_projections["verb"]["query"](verb_emb).unsqueeze(1)
        )
        target_q = self._split_heads(
            self.task_projections["target"]["query"](target_emb).unsqueeze(1)
        )

        # Create task-specific keys and values
        inst_k = self._split_heads(
            self.task_projections["instrument"]["key"](inst_emb).unsqueeze(1)
        )
        verb_k = self._split_heads(
            self.task_projections["verb"]["key"](verb_emb).unsqueeze(1)
        )
        target_k = self._split_heads(
            self.task_projections["target"]["key"](target_emb).unsqueeze(1)
        )

        inst_v = self._split_heads(
            self.task_projections["instrument"]["value"](inst_emb).unsqueeze(1)
        )
        verb_v = self._split_heads(
            self.task_projections["verb"]["value"](verb_emb).unsqueeze(1)
        )
        target_v = self._split_heads(
            self.task_projections["target"]["value"](target_emb).unsqueeze(1)
        )

        # Concatenate all keys and values
        all_keys = torch.cat([inst_k, verb_k, target_k], dim=2)
        all_values = torch.cat([inst_v, verb_v, target_v], dim=2)

        # Compute attention for each task's query against all keys/values
        inst_attn = self._scale_dot_product_attention(inst_q, all_keys, all_values)
        verb_attn = self._scale_dot_product_attention(verb_q, all_keys, all_values)
        target_attn = self._scale_dot_product_attention(target_q, all_keys, all_values)

        # Combine heads and squeeze the sequence dimension
        inst_attn = self._combine_heads(inst_attn).squeeze(1)
        verb_attn = self._combine_heads(verb_attn).squeeze(1)
        target_attn = self._combine_heads(target_attn).squeeze(1)

        # Project outputs
        inst_output = self.output_projections["instrument"](inst_attn)
        verb_output = self.output_projections["verb"](verb_attn)
        target_output = self.output_projections["target"](target_attn)

        # Apply dropout
        inst_output = self.dropout(inst_output)
        verb_output = self.dropout(verb_output)
        target_output = self.dropout(target_output)

        # Add residual connections and normalize
        inst_output = self.layer_norms["instrument"](inst_output + inst_residual)
        verb_output = self.layer_norms["verb"](verb_output + verb_residual)
        target_output = self.layer_norms["target"](target_output + target_residual)

        return inst_output, verb_output, target_output


class FFNLayer(nn.Module):
    """Feed-forward network layer inspired by Rendezvous FFN"""

    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x


class AttentionModule(nn.Module):
    """Attention module with explicit cross-task attention"""

    def __init__(self, feature_dims, num_layers=2, num_heads=4, common_dim=256):
        super().__init__()
        self.common_dim = common_dim

        # Project features to common dimensional space
        self.component_embeddings = nn.ModuleDict(
            {k: nn.Linear(v, common_dim) for k, v in feature_dims.items()}
        )

        # Stack of cross-task attention and FFN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CrossTaskAttention(common_dim, num_heads))
            self.layers.append(
                nn.ModuleDict(
                    {
                        "instrument": FFNLayer(common_dim),
                        "verb": FFNLayer(common_dim),
                        "target": FFNLayer(common_dim),
                    }
                )
            )

    def forward(self, verb_feat, inst_feat, target_feat):
        """
        Process features through the attention module

        Args:
            verb_feat: Verb features [batch_size, verb_dim]
            inst_feat: Instrument features [batch_size, inst_dim]
            target_feat: Target features [batch_size, target_dim]

        Returns:
            Processed features [batch_size, 3*common_dim]
        """
        # Project to common dimensional space
        verb_emb = self.component_embeddings["verb"](verb_feat)
        inst_emb = self.component_embeddings["instrument"](inst_feat)
        target_emb = self.component_embeddings["target"](target_feat)

        # Process through cross-task attention and FFN layers
        for i in range(0, len(self.layers), 2):
            # Cross-task attention layer
            inst_emb, verb_emb, target_emb = self.layers[i](
                inst_emb, verb_emb, target_emb
            )

            # Feed-forward layer for each task
            ffn_layer = self.layers[i + 1]
            inst_emb = ffn_layer["instrument"](inst_emb)
            verb_emb = ffn_layer["verb"](verb_emb)
            target_emb = ffn_layer["target"](target_emb)

        # Concatenate all embeddings for final output
        output = torch.cat([inst_emb, verb_emb, target_emb], dim=1)
        return output  # [batch_size, 3*common_dim]


class MultiTaskHead(nn.Module):
    """Classification head for each task (verb, instrument, target)"""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# class AttentionModule(nn.Module):  # 4th version
#     def __init__(self, feature_dims, num_layers=2, num_heads=4, common_dim=256):
#         super().__init__()
#         self.common_dim = common_dim
#         self.component_embeddings = nn.ModuleDict(
#             {k: nn.Linear(v, common_dim) for k, v in feature_dims.items()}
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             common_dim, num_heads, dim_feedforward=4 * common_dim, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

#     def forward(self, verb_feat, inst_feat, target_feat):
#         # Project features to common space
#         verb_emb = self.component_embeddings["verb"](verb_feat).unsqueeze(1)
#         inst_emb = self.component_embeddings["instrument"](inst_feat).unsqueeze(1)
#         target_emb = self.component_embeddings["target"](target_feat).unsqueeze(1)

#         # Concatenate as sequence tokens
#         tokens = torch.cat([verb_emb, inst_emb, target_emb], dim=1)

#         # Process through transformer
#         transformed = self.transformer(tokens)

#         # Flatten the sequence dimension
#         batch_size = transformed.size(0)
#         return transformed.reshape(batch_size, -1)


# class MHMA(nn.Module):
#     """
#     Multi-Head Mixed Attention module optimized for Swin3D features
#     Combines self-attention and cross-attention mechanisms in a streamlined way
#     """

#     def __init__(self, feature_dims, hidden_dim=256, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         # Create projection layers for each component
#         self.proj_layers = nn.ModuleDict(
#             {k: nn.Linear(v, hidden_dim) for k, v in feature_dims.items()}
#         )

#         # Query/key/value projections for cross-attention
#         self.q_proj = nn.Linear(hidden_dim * 3, hidden_dim)
#         self.k_proj = nn.ModuleDict(
#             {k: nn.Linear(hidden_dim, hidden_dim) for k in feature_dims.keys()}
#         )
#         self.v_proj = nn.ModuleDict(
#             {k: nn.Linear(hidden_dim, hidden_dim) for k in feature_dims.keys()}
#         )

#         # Self-attention for triplet features
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
#         )

#         # Output projection and normalization
#         self.output_proj = nn.Linear(hidden_dim * 4, hidden_dim * 3)
#         self.norm1 = nn.LayerNorm(hidden_dim * 3)
#         self.norm2 = nn.LayerNorm(hidden_dim * 3)
#         self.dropout = nn.Dropout(dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim * 3),
#         )

#     def forward(self, features):
#         # Unpack component features
#         instrument_feat = features["instrument"]
#         verb_feat = features["verb"]
#         target_feat = features["target"]

#         # Project each component to common dimension
#         inst_proj = self.proj_layers["instrument"](instrument_feat)
#         verb_proj = self.proj_layers["verb"](verb_feat)
#         target_proj = self.proj_layers["target"](target_feat)

#         # Concatenate for triplet representation
#         triplet_feat = torch.cat([inst_proj, verb_proj, target_proj], dim=1)

#         # Prepare for attention operations
#         batch_size = triplet_feat.size(0)

#         # Create query from triplet features
#         query = self.q_proj(triplet_feat).view(
#             batch_size, -1, self.num_heads, self.head_dim
#         )
#         query = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]

#         # Create keys and values for each component
#         keys = {
#             k: self.k_proj[k](v)
#             .view(batch_size, -1, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#             for k, v in {
#                 "instrument": inst_proj,
#                 "verb": verb_proj,
#                 "target": target_proj,
#             }.items()
#         }

#         values = {
#             k: self.v_proj[k](v)
#             .view(batch_size, -1, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#             for k, v in {
#                 "instrument": inst_proj,
#                 "verb": verb_proj,
#                 "target": target_proj,
#             }.items()
#         }

#         # Self-attention on triplet features
#         # Reshape for self-attention
#         triplet_for_self = triplet_feat.view(
#             batch_size, 3, -1
#         )  # [batch, 3, hidden_dim]
#         self_output, _ = self.self_attention(
#             triplet_for_self, triplet_for_self, triplet_for_self
#         )
#         self_output = self_output.reshape(batch_size, -1)  # [batch, 3*hidden_dim]

#         # Cross-attention between triplet query and component keys/values
#         def cross_attention(q, k, v):
#             # Scale dot-product attention
#             attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
#                 self.head_dim
#             )
#             attn_weights = F.softmax(attn_weights, dim=-1)
#             attn_weights = self.dropout(attn_weights)
#             return torch.matmul(attn_weights, v)

#         # Compute cross-attention for each component
#         inst_attended = cross_attention(query, keys["instrument"], values["instrument"])
#         verb_attended = cross_attention(query, keys["verb"], values["verb"])
#         target_attended = cross_attention(query, keys["target"], values["target"])

#         # Reshape and concatenate attention outputs
#         attended_outputs = [
#             x.transpose(1, 2).reshape(batch_size, -1)
#             for x in [inst_attended, verb_attended, target_attended]
#         ]

#         # Concatenate all attention outputs with self-attention
#         concat_attn = torch.cat([self_output] + attended_outputs, dim=1)

#         # Project to output dimension with residual connection
#         output = self.output_proj(concat_attn)
#         output = self.norm1(output + triplet_feat)

#         # Apply feed-forward network with residual connection
#         ff_output = self.ffn(output)
#         output = self.norm2(output + ff_output)

#         return output

# class AttentionModule(nn.Module):
#     def __init__(self, feature_dims, num_layers=2, num_heads=4, common_dim=256):
#         super().__init__()
#         self.common_dim = common_dim
#         self.feature_dims = feature_dims

#         # Initial projection to common dimension
#         self.projections = nn.ModuleDict({
#             k: nn.Sequential(
#                 nn.LayerNorm(v),
#                 nn.Linear(v, common_dim)
#             ) for k, v in feature_dims.items()
#         })

#         # Stack of MHMA layers
#         self.mhma_layers = nn.ModuleList([
#             MHMA(
#                 feature_dims={k: common_dim for k in feature_dims},
#                 hidden_dim=common_dim,
#                 num_heads=num_heads,
#                 dropout=0.1
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, verb_feat, inst_feat, target_feat):
#         # Project to common dimension
#         inst_proj = self.projections["instrument"](inst_feat)
#         verb_proj = self.projections["verb"](verb_feat)
#         target_proj = self.projections["target"](target_feat)

#         # Process through MHMA layers
#         features = {
#             "instrument": inst_proj,
#             "verb": verb_proj,
#             "target": target_proj
#         }

#         # Initial triplet representation
#         triplet_rep = torch.cat([inst_proj, verb_proj, target_proj], dim=1)

#         # Process through stacked MHMA layers
#         for mhma_layer in self.mhma_layers:
#             triplet_rep = mhma_layer(features)

#             # Update features for next layer
#             features = {
#                 "instrument": triplet_rep[:, :self.common_dim],
#                 "verb": triplet_rep[:, self.common_dim:2*self.common_dim],
#                 "target": triplet_rep[:, 2*self.common_dim:]
#             }

#         return triplet_rep


###################### FPN Implementation #############################


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block_module = Conv3dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
            )

            layer_block_module = Conv3dNormActivation(
                out_channels,
                out_channels,
                kernel_size=3,
                norm_layer=norm_layer,
                activation_layer=None,
            )

            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self.inner_blocks)
            for block in ["inner_blocks", "layer_blocks"]:
                for i in range(num_blocks):
                    for type in ["weight", "bias"]:
                        old_key = f"{prefix}{block}.{i}.{type}"
                        new_key = f"{prefix}{block}.{i}.0.{type}"
                        if old_key in state_dict:
                            state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = [self.get_result_from_layer_blocks(last_inner, -1)]

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            # Get all three spatial dimensions (D, H, W)
            feat_shape = inner_lateral.shape[-3:]

            # 3D upsampling
            inner_top_down = F.interpolate(
                last_inner,
                size=feat_shape,
                mode="nearest-exact",
            )

            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool3d  on top of the last feature map
    """

    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        # Use max pooling to simulate stride 2 subsampling
        x.append(F.max_pool3d(x[-1], kernel_size=1, stride=2, padding=0))
        return x, names


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            # print(module.__class__.__name__)
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                x_permuted = torch.permute(x, (0, 4, 1, 2, 3))
                out[out_name] = x_permuted  # [B, C, T, H, W]
                # print(x_permuted.shape)
        return out


# For adding FPN
class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


# For Combining backbone with FPN
class CustomSwin3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._create_backbone()
        self.out_channels = 256
        in_channels_list = [96, 192, 384, 768]

        backbone_module = nn.Sequential(
            OrderedDict(
                [
                    ("patch_embed", self.backbone.patch_embed),
                    ("pos_drop", self.backbone.pos_drop),
                    *[
                        (f"stage_{i}", layer)
                        for i, layer in enumerate(self.backbone.features)
                    ],
                ]
            )
        )

        # Modified return layers to capture features at the end of each main stage
        return_layers = {
            "stage_0": "0",  # After first transformer blocks (96 channels)
            "stage_2": "1",  # After second stage (192 channels)
            "stage_4": "2",  # After third stage (384 channels)
            "stage_6": "3",  # After final stage (768 channels)
        }

        self.model = BackboneWithFPN(
            backbone_module,
            return_layers,
            in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=None,
        )

    def forward(self, x):
        return self.model(x)

    def _create_backbone(self):
        backbone = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)
        # Remove the classification head components
        backbone.norm = nn.Identity()
        backbone.avgpool = nn.Identity()
        backbone.head = nn.Identity()

        # Rewrite the forward to skips the permute and flatten operations
        def forward(self, x: Tensor) -> Tensor:
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            x = self.features(x)
            x = self.norm(x)
            x = self.avgpool(x)
            x = self.head(x)
            return x

        backbone.forward = types.MethodType(forward, backbone)

        return backbone


# test_video = torch.randn(2, 3, 8, 224, 224)  # (B, C, T, H, W)
# model = CustomSwin3D()
# model = swin3d_s()
# outputs = model(test_video)
# print(outputs)

# print(outputs.shape)
