import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from utils import set_seeds, load_configs
import math
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dims, common_dim=128, num_heads=4):
        super().__init__()
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.head_dim = common_dim // num_heads

        # Projections for each feature type
        self.verb_proj = nn.Linear(feature_dims["verb"], common_dim * 3)
        self.inst_proj = nn.Linear(feature_dims["instrument"], common_dim * 3)
        self.target_proj = nn.Linear(feature_dims["target"], common_dim * 3)

        self.output_proj = nn.Linear(common_dim * 3, common_dim)

    def forward(self, verb_feat, inst_feat, target_feat):
        B = verb_feat.size(0)

        # Project each feature type to Q, K, V
        verb_q, verb_k, verb_v = self.verb_proj(verb_feat).chunk(3, dim=-1)
        inst_q, inst_k, inst_v = self.inst_proj(inst_feat).chunk(3, dim=-1)
        target_q, target_k, target_v = self.target_proj(target_feat).chunk(3, dim=-1)

        # Reshape for multi-head attention
        def reshape(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention for each combination
        def attention(q, k, v):
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            return (attn @ v).transpose(1, 2).reshape(B, -1)

        # Compute three-way attention
        verb_out = attention(
            reshape(verb_q),
            torch.cat([reshape(inst_k), reshape(target_k)], dim=-2),
            torch.cat([reshape(inst_v), reshape(target_v)], dim=-2),
        )

        inst_out = attention(
            reshape(inst_q),
            torch.cat([reshape(verb_k), reshape(target_k)], dim=-2),
            torch.cat([reshape(verb_v), reshape(target_v)], dim=-2),
        )

        target_out = attention(
            reshape(target_q),
            torch.cat([reshape(verb_k), reshape(inst_k)], dim=-2),
            torch.cat([reshape(verb_v), reshape(inst_v)], dim=-2),
        )

        # Combine and project
        combined = torch.cat([verb_out, inst_out, target_out], dim=-1)
        return self.output_proj(combined)


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


class ModelValidator:
    def __init__(
        self,
        val_loader,
        num_classes: dict,
        device: str,
        model_path: str,
    ):
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.model_path = model_path
        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self.cross_attention = CrossAttentionFusion(self.feature_dims).to(self.device)
        # Initialize and load the model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model architecture and load the saved weights"""
        model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        in_features = model.num_features
        # Teacher heads
        model.verb_head = MultiTaskHead(in_features, self.num_classes["verb"]).to(
            self.device
        )
        model.instrument_head = MultiTaskHead(
            in_features, self.num_classes["instrument"]
        ).to(self.device)
        model.target_head = MultiTaskHead(in_features, self.num_classes["target"]).to(
            self.device
        )

        # Teacher triplet head combines the ivt heads output features
        common_dim = self.cross_attention.common_dim
        total_input_size = in_features + sum(self.feature_dims.values()) + common_dim
        model.triplet_head = nn.Sequential(
            nn.LayerNorm(total_input_size),
            nn.Dropout(p=0.5),
            nn.Linear(total_input_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.num_classes["triplet"]),
        ).to(self.device)
        # We need to remove the original classification head
        model.head = nn.Identity()

        # Load the saved weights
        print(f"Loading model from {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        return model

    def _forward_pass(self, inputs, mode="teacher"):
        """Perform forward pass through the model based on the model type [teacher, student]"""
        features = self.model(inputs)
        if mode == "teacher":
            # Get individual component predictions
            verb_logits = self.model.verb_head(features)
            instrument_logits = self.model.instrument_head(features)
            target_logits = self.model.target_head(features)
            # Apply cross-attention fusion
            attention_output = self.cross_attention(
                verb_logits, instrument_logits, target_logits
            )
            # Combine all features for triplet prediction
            combined_features = torch.cat(
                [
                    features,  # original features from backbone
                    verb_logits,  # verb predictions
                    instrument_logits,  # instrument predictions
                    target_logits,  # target predictions
                    attention_output,  # attention-fused features
                ],
                dim=1,
            )
            triplet_logits = self.model.triplet_head(combined_features)

            return {
                "verb": verb_logits,
                "instrument": instrument_logits,
                "target": target_logits,
                "triplet": triplet_logits,
            }

    def validate(self):
        """Validate the model and compute metrics"""
        self.model.eval()

        # Initialize ivtmetrics Recognition object
        recognize = Recognition(num_class=self.num_classes["triplet"])
        recognize.reset_global()

        with torch.no_grad():
            for inputs, batch_labels in self.val_loader:
                inputs = inputs.to(self.device)
                model_outputs = self._forward_pass(inputs)
                predictions = torch.sigmoid(model_outputs["triplet"]).cpu().numpy()
                true_labels = batch_labels["triplet"].cpu().numpy()
                recognize.update(true_labels, predictions)

        # Compute and log final metrics
        results = {
            "triplet": recognize.compute_global_AP("ivt"),
            "verb": recognize.compute_global_AP("v"),
            "instrument": recognize.compute_global_AP("i"),
            "target": recognize.compute_global_AP("t"),
        }

        print("\nValidation Results:")
        for task in ["verb", "instrument", "target", "triplet"]:
            print(f"{task.capitalize()} Results:")
            print(f"mAP: {results[task]['mAP']:.4f}")
            print(f"AP: {results[task]['AP']}")


def main():
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    ANNOTATIONS_PATH = r"05_datasets_dir/CholecT50/annotations.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"
    MODEL_PATH = (
        r"04_models_dir/training_20250205_234859/best_model_triplet_teacher.pth"
    )

    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda:1")
    set_seeds()
    configs = load_configs(CONFIGS_PATH)

    val_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="val",
        train_ratio=configs["train_ratio"],
        train=False,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
    )

    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], shuffle=False
    )
    # Define your model configuration
    num_classes = {
        "verb": 10,
        "instrument": 6,
        "target": 15,
        "triplet": 69,
    }
    # Initialize validator
    validator = ModelValidator(
        val_loader=val_loader,
        num_classes=num_classes,
        device=DEVICE,
        model_path=MODEL_PATH,
    )

    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
