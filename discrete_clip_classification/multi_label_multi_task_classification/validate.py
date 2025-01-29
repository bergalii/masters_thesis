import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from ivtmetrics import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from utils import set_seeds


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dims, common_dim=128):
        super().__init__()
        self.common_dim = common_dim
        self.query = nn.Linear(feature_dims["verb"], common_dim)
        self.key = nn.Linear(feature_dims["instrument"], common_dim)
        self.value = nn.Linear(feature_dims["target"], common_dim)

    def forward(self, verb_feat, inst_feat, target_feat):
        # Project features to common dimension
        Q = self.query(verb_feat)  # (B, common_dim)
        K = self.key(inst_feat)  # (B, common_dim)
        V = self.value(target_feat)  # (B, common_dim)

        # Compute scaled dot-product attention
        scale_factor = 1.0 / (self.common_dim**0.5)
        attention_scores = (
            torch.bmm(Q.unsqueeze(1), K.unsqueeze(1).transpose(1, 2)) * scale_factor
        )
        attention = torch.softmax(attention_scores, dim=-1)

        # Apply attention to value
        attention_output = torch.bmm(attention, V.unsqueeze(1)).squeeze(
            1
        )  # (B, common_dim)
        return attention_output


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
        self.feature_dims = {
            "verb": self.num_classes["verb"],
            "instrument": self.num_classes["instrument"],
            "target": self.num_classes["target"],
        }
        self.cross_attention = CrossAttentionFusion(self.feature_dims).to(self.device)
        # Initialize and load the model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model architecture and load the saved weights"""
        model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)

        in_features = model.num_features

        # Create separate heads for each task
        model.verb_head = (MultiTaskHead(in_features, self.num_classes["verb"])).to(
            self.device
        )

        model.instrument_head = MultiTaskHead(
            in_features, self.num_classes["instrument"]
        ).to(self.device)

        model.target_head = MultiTaskHead(in_features, self.num_classes["target"]).to(
            self.device
        )

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

        # Remove original classification head
        model.head = nn.Identity()

        # Load the saved weights
        print(f"Loading model from {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        return model

    def _forward_pass(self, inputs):
        """Perform forward pass through the model and all task heads"""
        features = self.model(inputs)
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

                # Process triplet predictions
                predictions = torch.sigmoid(model_outputs["triplet"]).cpu().numpy()
                true_labels = batch_labels["triplet"].cpu().numpy()

                # Update metrics
                recognize.update(true_labels, predictions)
                recognize.video_end()

        # Compute and log final metrics
        results = {}
        results["instrument"] = recognize.compute_video_AP("i")
        results["verb"] = recognize.compute_video_AP("v")
        results["target"] = recognize.compute_video_AP("t")
        results["triplet"] = recognize.compute_video_AP("ivt")

        print("\nValidation Results:")
        for task in ["verb", "instrument", "target", "triplet"]:
            print(f"{task.capitalize()} Results:")
            print(f"mAP: {results[task]['mAP']:.4f}")
            # print(f"AP: {results[task]['AP']}")


def main():
    CLIPS_DIR = r"videos"
    ANNOTATIONS_PATH = r"annotations.csv"
    MODEL_PATH = r"models/training_20250128_225758/best_model_triplet_True.pth"
    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda:1")
    set_seeds()

    batch_size = 8

    val_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=10,
        split="val",
        train=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define your model configuration
    num_classes = {
        "verb": 10,
        "instrument": 6,
        "target": 15,
        "triplet": 100,
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
