import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from utils import set_seeds, load_configs


class TripletHead(nn.Module):
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


class SimplifiedModelValidator:
    def __init__(
        self,
        val_loader,
        val_dataset,
        num_classes: dict,
        device: str,
        model_path: str,
        hidden_layer_dim: int,
    ):
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = device
        self.model_path = model_path
        self.hidden_layer_dim = hidden_layer_dim
        # Initialize and load the model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the simplified model architecture and load the saved weights"""
        model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        # Remove the original classification head
        model.head = nn.Identity()
        in_features = model.num_features

        # Add simplified triplet head only - matching the TripletHead architecture in SimplifiedTrainer
        model.triplet_head = TripletHead(
            in_features, self.num_classes["triplet"], self.hidden_layer_dim
        ).to(self.device)

        # Load the saved weights
        print(f"Loading model from {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return model

    def _forward_pass(self, inputs):
        """Perform simple forward pass through the backbone and triplet head only"""
        backbone_features = self.model(inputs)
        triplet_logits, _ = self.model.triplet_head(backbone_features)
        return triplet_logits

    def validate(self):
        """Validate the model and compute metrics"""
        self.model.eval()

        # Initialize recognition object
        recognize = Recognition(num_class=self.num_classes["triplet"])
        recognize.reset()

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                # inputs have dimension [B, N, C, T, H, W]
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]
                # Process each video in the batch
                for b in range(batch_size):
                    video_outputs = []
                    # Process each clip individually
                    for c in range(num_clips):
                        # Extract single clip: [C, T, H, W]
                        clip = (
                            inputs_batch[b, c].unsqueeze(0).to(self.device)
                        )  # Add batch dimension

                        # Get the predictions for this clip
                        clip_logits = self._forward_pass(clip)
                        video_outputs.append(clip_logits)

                    # Average predictions across clips
                    outputs_tensor = torch.cat(video_outputs, dim=0)
                    avg_logits = torch.mean(outputs_tensor, dim=0, keepdim=True)

                    # Convert to probabilities
                    predictions = torch.sigmoid(avg_logits).cpu().numpy()
                    labels = batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()

                    # Update the recognizer with the current video
                    recognize.update(labels, predictions)
                    # Signal end of video to enable video-level metrics
                    recognize.video_end()

        # Compute video-level AP metrics
        results = {
            "triplet": recognize.compute_video_AP("ivt"),
            "verb": recognize.compute_video_AP("v"),
            "instrument": recognize.compute_video_AP("i"),
            "target": recognize.compute_video_AP("t"),
        }

        print("\nVideo-level Validation Results:")
        for task in ["verb", "instrument", "target", "triplet"]:
            print(f"{task.capitalize()} Results:")
            print(f"Video-level mAP: {results[task]['mAP']:.4f}")
            print(f"Video-level AP: {results[task]['AP']}")

        # Compute global-level AP metrics
        global_results = {
            "triplet": recognize.compute_AP("ivt"),
            "verb": recognize.compute_AP("v"),
            "instrument": recognize.compute_AP("i"),
            "target": recognize.compute_AP("t"),
        }

        print("-" * 50)
        print("\nGlobal Validation Results:")
        for task in ["verb", "instrument", "target", "triplet"]:
            print(f"{task.capitalize()} Results:")
            print(f"Global mAP: {global_results[task]['mAP']:.4f}")

        # If you need more detailed class information for triplets
        print("\nDetailed Triplet Class Information:")
        for i, ap in enumerate(global_results["triplet"]["AP"]):
            original_id = getattr(self.val_dataset, "index_to_triplet", {}).get(i, i)
            print(f"  Class {original_id}: AP = {ap:.4f}")

        return results, global_results


def main():
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    ANNOTATIONS_PATH = r"05_datasets_dir/CholecT50/annotations.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"
    MODEL_PATH = r"04_models_dir/training_20250428_094504/best_model.pth"

    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda:1")
    set_seeds()
    configs = load_configs(CONFIGS_PATH)
    CROSS_VAL_FOLD = 2

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
        cross_val_fold=CROSS_VAL_FOLD,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], shuffle=False
    )

    # Initialize validator with simplified model structure
    validator = SimplifiedModelValidator(
        val_loader=val_loader,
        val_dataset=val_dataset,
        num_classes=val_dataset.num_classes,
        device=DEVICE,
        model_path=MODEL_PATH,
        hidden_layer_dim=configs["hidden_layer_dim"],
    )
    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
