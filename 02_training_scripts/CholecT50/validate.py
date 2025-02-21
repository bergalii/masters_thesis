import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from utils import set_seeds, load_configs
from modules import MultiTaskHead, AttentionModule


class ModelValidator:
    def __init__(
        self,
        val_loader,
        num_classes: dict,
        device: str,
        model_path: str,
        triplet_to_ivt: dict,
        guidance_scale: float = 0.8,
    ):
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.model_path = model_path
        self.guidance_scale = guidance_scale
        self.triplet_to_ivt = torch.tensor(
            [triplet_to_ivt[idx] for idx in range(len(triplet_to_ivt))],
            device=device,
        )
        self.MI = torch.zeros(
            (self.num_classes["instrument"], self.num_classes["triplet"])
        ).to(device)
        self.MV = torch.zeros(
            (self.num_classes["verb"], self.num_classes["triplet"])
        ).to(device)
        self.MT = torch.zeros(
            (self.num_classes["target"], self.num_classes["triplet"])
        ).to(device)
        for t, (inst, verb, target) in triplet_to_ivt.items():
            self.MI[inst, t] = 1
            self.MV[verb, t] = 1
            self.MT[target, t] = 1

        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self.cross_attention = AttentionModule(self.feature_dims).to(self.device)
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
        total_input_size = (
            in_features + sum(self.feature_dims.values()) + 3 * common_dim
        )
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
                # Convert all outputs to probabilities at once
                task_probabilities = {
                    task: torch.sigmoid(outputs)
                    for task, outputs in model_outputs.items()
                }

                guidance_inst = torch.matmul(task_probabilities["instrument"], self.MI)
                guidance_verb = torch.matmul(task_probabilities["verb"], self.MV)
                guidance_target = torch.matmul(task_probabilities["target"], self.MT)

                # Combine guidance signals
                guidance = guidance_inst * guidance_verb * guidance_target

                # Apply guidance with a scale factor
                guided_probs = (1 - self.guidance_scale) * task_probabilities[
                    "triplet"
                ] + self.guidance_scale * (guidance * task_probabilities["triplet"])

                predictions = guided_probs.cpu().numpy()
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
        r"04_models_dir/training_20250217_103632/best_model_triplet_teacher.pth"
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
        triplet_to_ivt=val_dataset.triplet_to_ivt,
    )

    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
