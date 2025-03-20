import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from utils import set_seeds, load_configs
from modules import MultiTaskHead, AttentionModule
from collections import defaultdict


class ModelValidator:
    def __init__(
        self,
        val_loader,
        val_dataset,
        num_classes: dict,
        device: str,
        model_path: str,
        triplet_to_ivt: dict,
        guidance_scale: float,
    ):
        self.val_loader = val_loader
        self.val_dataset = val_dataset
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

        # self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}
        self.feature_dims = {
            "verb": 512,
            "instrument": 512,
            "target": 512,
        }
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

        model.attention_module = AttentionModule(self.feature_dims).to(self.device)

        # Teacher triplet head combines the ivt heads output features
        common_dim = model.attention_module.common_dim
        total_input_size = in_features + 3 * common_dim
        model.triplet_head = MultiTaskHead(
            total_input_size, self.num_classes["triplet"]
        ).to(self.device)
        model.head = nn.Identity()

        # Load the saved weights
        print(f"Loading model from {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        return model

    def _forward_pass(self, inputs):
        """Perform forward pass through the model"""
        features = self.model(inputs)
        # Get individual component predictions
        verb_logits = self.model.verb_head(features)
        instrument_logits = self.model.instrument_head(features)
        target_logits = self.model.target_head(features)
        # Apply attention module
        attention_output = self.model.attention_module(
            verb_logits, instrument_logits, target_logits
        )
        # Combine all features for triplet prediction
        combined_features = torch.cat(
            [
                features,  # original features from backbone
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

        # Get all unique video IDs
        video_ids = list(set([item["video_id"] for _, item in self.val_dataset]))

        # Initialize recognition object
        recognize = Recognition(num_class=self.num_classes["triplet"])
        recognize.reset_global()

        # Group samples by video_id
        video_samples = defaultdict(list)
        for i in range(len(self.val_dataset)):
            _, labels = self.val_dataset[i]
            video_id = labels["video_id"]
            video_samples[video_id].append(i)

        with torch.no_grad():
            # Process each video separately
            for video_id in video_ids:
                # Process all clips from this video
                for idx in video_samples[video_id]:
                    inputs, batch_labels = self.val_dataset[idx]
                    # Add batch dimension
                    inputs = inputs.unsqueeze(0).to(self.device)

                    model_outputs = self._forward_pass(inputs)
                    # Convert all outputs to probabilities at once
                    task_probabilities = {
                        task: torch.sigmoid(outputs)
                        for task, outputs in model_outputs.items()
                    }

                    guidance_inst = torch.matmul(
                        task_probabilities["instrument"], self.MI
                    )
                    guidance_verb = torch.matmul(task_probabilities["verb"], self.MV)
                    guidance_target = torch.matmul(
                        task_probabilities["target"], self.MT
                    )

                    # Combine guidance signals
                    guidance = guidance_inst * guidance_verb * guidance_target

                    # Apply guidance with a scale factor
                    guided_probs = (1 - self.guidance_scale) * task_probabilities[
                        "triplet"
                    ] + self.guidance_scale * (guidance * task_probabilities["triplet"])

                    predictions = guided_probs.cpu().numpy()
                    true_labels = batch_labels["triplet"].unsqueeze(0).cpu().numpy()
                    recognize.update(true_labels, predictions)

                # Signal end of video to the Recognition evaluator
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

        global_results = {
            "triplet": recognize.compute_global_AP("ivt"),
            "verb": recognize.compute_global_AP("v"),
            "instrument": recognize.compute_global_AP("i"),
            "target": recognize.compute_global_AP("t"),
        }

        print("-" * 50)
        print("\nGlobal Validation Results (for comparison):")
        for task in ["verb", "instrument", "target", "triplet"]:
            print(f"{task.capitalize()} Results:")
            print(f"Global mAP: {global_results[task]['mAP']:.4f}")


def main():
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    ANNOTATIONS_PATH = r"05_datasets_dir/CholecT50/annotations.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"
    MODEL_PATH = r"04_models_dir/training_20250311_202110/best_model_teacher.pth"

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

    # Initialize validator
    validator = ModelValidator(
        val_loader=val_loader,
        val_dataset=val_dataset,
        num_classes=val_dataset.num_classes,
        device=DEVICE,
        model_path=MODEL_PATH,
        triplet_to_ivt=val_dataset.triplet_to_ivt,
        guidance_scale=configs["guidance_scale"],
    )

    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
