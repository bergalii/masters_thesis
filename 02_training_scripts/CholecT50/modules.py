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


# class AttentionModule(nn.Module):
#     def __init__(
#         self, feature_dims, common_dim, num_layers=2, num_heads=8, dropout=0.1
#     ):
#         super().__init__()
#         self.common_dim = common_dim

#         # Project each component's features to a common dimension
#         self.component_projections = nn.ModuleDict(
#             {
#                 k: nn.Sequential(
#                     nn.Linear(v, common_dim),
#                     nn.LayerNorm(common_dim),
#                     nn.Dropout(dropout),
#                 )
#                 for k, v in feature_dims.items()
#             }
#         )

#         # Cross-attention layers between components
#         self.cross_attention = nn.ModuleDict(
#             {
#                 # Each component attends to others
#                 "verb_to_others": nn.MultiheadAttention(
#                     common_dim, num_heads, dropout=dropout, batch_first=True
#                 ),
#                 "instrument_to_others": nn.MultiheadAttention(
#                     common_dim, num_heads, dropout=dropout, batch_first=True
#                 ),
#                 "target_to_others": nn.MultiheadAttention(
#                     common_dim, num_heads, dropout=dropout, batch_first=True
#                 ),
#             }
#         )

#         # Layer norms for pre and post attention
#         self.pre_cross_norm = nn.ModuleDict(
#             {k: nn.LayerNorm(common_dim) for k in feature_dims.keys()}
#         )
#         self.post_cross_norm = nn.ModuleDict(
#             {k: nn.LayerNorm(common_dim) for k in feature_dims.keys()}
#         )

#         # Add a "global token" that can attend to all components
#         self.global_token = nn.Parameter(torch.randn(1, 1, common_dim) * 0.02)

#         # Create transformer encoder for sequence processing
#         encoder_layer = nn.TransformerEncoderLayer(
#             common_dim,
#             num_heads,
#             dim_feedforward=4 * common_dim,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,  # Pre-norm architecture for better training stability
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

#         # Final output processing
#         self.output_projection = nn.Sequential(
#             nn.Linear(
#                 4 * common_dim, 3 * common_dim
#             ),  # 4 = 3 components + global token
#             nn.LayerNorm(3 * common_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#     def forward(self, verb_feat, inst_feat, target_feat):
#         batch_size = verb_feat.size(0)

#         # Project inputs to common dimension
#         verb_proj = self.component_projections["verb"](verb_feat)
#         inst_proj = self.component_projections["instrument"](inst_feat)
#         target_proj = self.component_projections["target"](target_feat)

#         # Reshape for attention - make sure they have a sequence dimension
#         verb_seq = verb_proj.unsqueeze(1)  # [batch, 1, common_dim]
#         inst_seq = inst_proj.unsqueeze(1)  # [batch, 1, common_dim]
#         target_seq = target_proj.unsqueeze(1)  # [batch, 1, common_dim]

#         # Apply cross-attention with residual connections
#         # Verb attends to instrument and target
#         norm_verb = self.pre_cross_norm["verb"](verb_seq)
#         kv_for_verb = torch.cat([inst_seq, target_seq], dim=1)
#         verb_cross, _ = self.cross_attention["verb_to_others"](
#             norm_verb, kv_for_verb, kv_for_verb
#         )
#         verb_cross = verb_seq + verb_cross  # Residual connection
#         verb_cross = self.post_cross_norm["verb"](verb_cross)

#         # Instrument attends to verb and target
#         norm_inst = self.pre_cross_norm["instrument"](inst_seq)
#         kv_for_inst = torch.cat([verb_seq, target_seq], dim=1)
#         inst_cross, _ = self.cross_attention["instrument_to_others"](
#             norm_inst, kv_for_inst, kv_for_inst
#         )
#         inst_cross = inst_seq + inst_cross  # Residual connection
#         inst_cross = self.post_cross_norm["instrument"](inst_cross)

#         # Target attends to verb and instrument
#         norm_target = self.pre_cross_norm["target"](target_seq)
#         kv_for_target = torch.cat([verb_seq, inst_seq], dim=1)
#         target_cross, _ = self.cross_attention["target_to_others"](
#             norm_target, kv_for_target, kv_for_target
#         )
#         target_cross = target_seq + target_cross  # Residual connection
#         target_cross = self.post_cross_norm["target"](target_cross)

#         # Expand global token to batch size
#         global_token = self.global_token.expand(batch_size, -1, -1)

#         # Concatenate all tokens (including global token) for transformer processing
#         all_tokens = torch.cat(
#             [global_token, verb_cross, inst_cross, target_cross],  # Global token first
#             dim=1,
#         )  # [batch, 4, common_dim]

#         # Process through transformer encoder
#         transformed = self.transformer(all_tokens)

#         # Reshape and project to final output
#         batch_size = transformed.size(0)
#         flattened = transformed.reshape(batch_size, -1)

#         return self.output_projection(flattened)


class AttentionModule(nn.Module):
    def __init__(self, feature_dims, common_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.common_dim = common_dim
        self.component_embeddings = nn.ModuleDict(
            {k: nn.Linear(v, common_dim) for k, v in feature_dims.items()}
        )

        encoder_layer = nn.TransformerEncoderLayer(
            common_dim, num_heads, dim_feedforward=4 * common_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, verb_feat, inst_feat, target_feat):
        # Project features to common space
        verb_emb = self.component_embeddings["verb"](verb_feat).unsqueeze(1)
        inst_emb = self.component_embeddings["instrument"](inst_feat).unsqueeze(1)
        target_emb = self.component_embeddings["target"](target_feat).unsqueeze(1)

        # Concatenate as sequence tokens
        tokens = torch.cat([verb_emb, inst_emb, target_emb], dim=1)

        # Process through transformer
        transformed = self.transformer(tokens)

        # Flatten the sequence dimension
        batch_size = transformed.size(0)
        return transformed.reshape(batch_size, -1)


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_layer_dim: int):
        super().__init__()
        # Extract hidden features for attention
        self.hidden = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, hidden_layer_dim),
            nn.GELU(),
        )
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_layer_dim, num_classes),
        )

    def forward(self, x):
        hidden_features = self.hidden(x)
        logits = self.classifier(hidden_features)
        return logits, hidden_features


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
