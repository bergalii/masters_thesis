import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from utils import set_seeds, load_configs
from modules import MultiTaskHead, AttentionModule
import numpy as np


class ModelValidator:
    def __init__(
        self,
        val_loader,
        val_dataset,
        num_classes: dict,
        device: str,
        model_path: str,
        triplet_to_ivt: dict,
        attention_module_common_dim: int,
        hidden_layer_dim: int,
        guidance_scale: float,
        label_mappings: dict,
    ):
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = device
        self.model_path = model_path
        self.guidance_scale = guidance_scale
        self.label_mappings = label_mappings
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
        self.hidden_layer_dim = hidden_layer_dim
        self.attention_module_common_dim = attention_module_common_dim
        self.feature_dims = {k: v for k, v in num_classes.items() if k != "triplet"}

        # Initialize and load the model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model architecture and load the saved weights"""
        model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        # We need to remove the original classification head
        model.head = nn.Identity()
        in_features = model.num_features

        # Teacher heads
        model.verb_head = MultiTaskHead(
            in_features, self.num_classes["verb"], self.hidden_layer_dim
        ).to(self.device)
        model.instrument_head = MultiTaskHead(
            in_features, self.num_classes["instrument"], self.hidden_layer_dim
        ).to(self.device)
        model.target_head = MultiTaskHead(
            in_features, self.num_classes["target"], self.hidden_layer_dim
        ).to(self.device)

        model.attention_module = AttentionModule(
            self.feature_dims,
            self.hidden_layer_dim,
            self.attention_module_common_dim,
            num_heads=4,
            dropout=0.3,
        ).to(self.device)

        total_input_size = (
            in_features  # Backbone features
            + self.attention_module_common_dim  # Attention module output
            + sum(self.feature_dims.values())  # Probability outputs from each tas
        )

        model.triplet_head = MultiTaskHead(
            total_input_size, self.num_classes["triplet"], self.hidden_layer_dim
        ).to(self.device)

        # Load the saved weights
        print(f"Loading model from {self.model_path}")
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        return model

    def _forward_pass(self, inputs):
        """Perform forward pass through the model based on the model type [teacher, student]"""
        backbone_features = self.model(inputs)
        # Get individual component predictions
        verb_logits, verb_hidden = self.model.verb_head(backbone_features)
        instrument_logits, inst_hidden = self.model.instrument_head(backbone_features)
        target_logits, target_hidden = self.model.target_head(backbone_features)

        attention_output = self.model.attention_module(
            verb_hidden,
            verb_logits,
            inst_hidden,
            instrument_logits,
            target_hidden,
            target_logits,
        )

        # Combine all features for triplet prediction
        combined_features = torch.cat(
            [
                backbone_features,  # original features from backbone
                attention_output,  # attention-fused features
                torch.sigmoid(verb_logits),  # Probability predictions
                torch.sigmoid(instrument_logits),
                torch.sigmoid(target_logits),
            ],
            dim=1,
        )

        triplet_logits, _ = self.model.triplet_head(combined_features)

        return {
            "verb": verb_logits,
            "instrument": instrument_logits,
            "target": target_logits,
            "triplet": triplet_logits,
        }

    def validate(self):
        """Validate model and compute metrics for all tasks using sklearn"""

        self.model.eval()

        # Initialize storage for predictions and labels
        all_predictions = {
            task: [] for task in ["verb", "instrument", "target", "triplet"]
        }
        all_labels = {task: [] for task in ["verb", "instrument", "target", "triplet"]}

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                # inputs have dimension [B, N, C, T, H, W]
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]

                # Process each video in the batch
                for b in range(batch_size):
                    video_outputs = {
                        task: [] for task in ["verb", "instrument", "target", "triplet"]
                    }
                    # Process each clip individually
                    for c in range(num_clips):
                        # Extract single clip: [C, T, H, W]
                        clip = (
                            inputs_batch[b, c].unsqueeze(0).to(self.device)
                        )  # Add batch dimension

                        # Get the predictions for this clip
                        clip_outputs = self._forward_pass(clip)

                        # Store predictions for each task
                        for task, outputs in clip_outputs.items():
                            video_outputs[task].append(outputs)

                    # Average predictions across clips for each task
                    task_logits = {}
                    for task, outputs_list in video_outputs.items():
                        # Concatenate along batch dimension then average
                        outputs_tensor = torch.cat([o for o in outputs_list], dim=0)
                        task_logits[task] = torch.mean(
                            outputs_tensor, dim=0, keepdim=True
                        )

                    # Convert all task logits to probabilities
                    task_probabilities = {
                        task: torch.sigmoid(logits)
                        for task, logits in task_logits.items()
                    }

                    # Get guidance from individual tasks
                    guidance_inst = torch.matmul(
                        task_probabilities["instrument"], self.MI
                    )
                    guidance_verb = torch.matmul(task_probabilities["verb"], self.MV)
                    guidance_target = torch.matmul(
                        task_probabilities["target"], self.MT
                    )

                    # Combine guidance outputs
                    guidance = guidance_inst * guidance_verb * guidance_target

                    # Apply guidance with a scale factor
                    guided_triplet_probs = (
                        1 - self.guidance_scale
                    ) * task_probabilities["triplet"] + self.guidance_scale * (
                        guidance * task_probabilities["triplet"]
                    )

                    # Store predictions and labels for each task
                    for task in ["verb", "instrument", "target"]:
                        all_predictions[task].append(
                            task_probabilities[task].cpu().numpy()
                        )
                        all_labels[task].append(
                            batch_labels[task][b].unsqueeze(0).cpu().numpy()
                        )

                    # Use guided predictions for triplet
                    all_predictions["triplet"].append(
                        guided_triplet_probs.cpu().numpy()
                    )
                    all_labels["triplet"].append(
                        batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()
                    )

        # Convert lists to numpy arrays
        for task in ["verb", "instrument", "target", "triplet"]:
            all_predictions[task] = np.vstack(all_predictions[task])
            all_labels[task] = np.vstack(all_labels[task])

        # Compute metrics
        task_metrics = {}

        for task in ["verb", "instrument", "target", "triplet"]:
            predictions = all_predictions[task]
            labels = all_labels[task]

            # Use Average Precision (AP) as primary metric
            class_aps = []
            class_precisions = []
            class_recalls = []
            class_f1s = []

            for i in range(predictions.shape[1]):
                class_preds = predictions[:, i]
                class_labels = labels[:, i]

                # Skip if no positive samples
                if np.sum(class_labels) == 0:
                    class_aps.append(0.0)
                    class_precisions.append(0.0)
                    class_recalls.append(0.0)
                    class_f1s.append(0.0)
                    continue

                # Calculate Average Precision (equivalent to mAP per class)
                ap = average_precision_score(class_labels, class_preds)
                class_aps.append(ap)

                # Calculate metrics with fixed threshold (0.5) for consistency
                binary_preds = (class_preds > 0.5).astype(int)
                precision = precision_score(class_labels, binary_preds, zero_division=0)
                recall = recall_score(class_labels, binary_preds, zero_division=0)
                f1 = f1_score(class_labels, binary_preds, zero_division=0)

                class_precisions.append(precision)
                class_recalls.append(recall)
                class_f1s.append(f1)

            # Calculate mean metrics (only from classes with positive samples)
            valid_classes = [
                i for i in range(len(class_aps)) if np.sum(labels[:, i]) > 0
            ]
            mean_ap = (
                np.mean([class_aps[i] for i in valid_classes]) if valid_classes else 0.0
            )
            mean_precision = (
                np.mean([class_precisions[i] for i in valid_classes])
                if valid_classes
                else 0.0
            )
            mean_recall = (
                np.mean([class_recalls[i] for i in valid_classes])
                if valid_classes
                else 0.0
            )
            mean_f1 = (
                np.mean([class_f1s[i] for i in valid_classes]) if valid_classes else 0.0
            )

            # Store metrics
            task_metrics[task] = {
                "mAP": mean_ap,
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": mean_f1,
                "class_aps": class_aps,
                "class_precisions": class_precisions,
                "class_recalls": class_recalls,
                "class_f1s": class_f1s,
            }

            # Print the results
            print(f"{task.upper()} METRICS:")
            print(f"  mAP: {mean_ap:.4f}")
            print(f"  Precision: {mean_precision:.4f}")
            print(f"  Recall: {mean_recall:.4f}")
            print(f"  F1-Score: {mean_f1:.4f}")

            # Print per-class AP metrics
            print(f"\n{task.upper()} PER-CLASS AP:")
            for i in range(len(class_aps)):
                # Get the class name based on the component
                label_name = self.label_mappings[task].get(i, f"Class_{i}")
                print(f"  {label_name}: {class_aps[i]:.4f}")

            print("-" * 50)

        return task_metrics


def main():
    CLIPS_DIR = r"/data/Berk/masters_thesis/05_datasets_dir/UKE/clips"
    ANNOTATIONS_PATH = r"/data/Berk/masters_thesis/annotations_combined.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"
    MODEL_PATH = r"04_models_dir/training_20250617_225106_UKE/best_model_teacher.pth"

    torch.cuda.set_device(1)
    DEVICE = torch.device("cuda:1")
    set_seeds()
    configs = load_configs(CONFIGS_PATH)

    train_dataset = MultiTaskVideoDataset(
        clips_dir=CLIPS_DIR,
        annotations_path=ANNOTATIONS_PATH,
        clip_length=configs["clip_length"],
        split="train",
        train_ratio=configs["train_ratio"],
        train=True,
        frame_width=configs["frame_width"],
        frame_height=configs["frame_height"],
        min_occurrences=configs["min_occurrences"],
    )
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

    # Create label mappings
    label_mappings = train_dataset.label_mappings

    # ACTIVE_TRIPLETS = [
    #     2,
    #     3,
    #     5,
    #     7,
    #     8,
    #     9,
    #     11,
    #     12,
    #     13,
    #     14,
    #     15,
    #     16,
    #     18,
    #     19,
    #     20,
    #     22,
    #     23,
    #     24,
    #     26,
    #     27,
    #     28,
    #     29,
    #     31,
    #     32,
    #     35,
    #     39,
    #     40,
    #     41,
    #     42,
    #     43,
    #     44,
    #     45,
    #     46,
    #     47,
    #     48,
    #     49,
    #     50,
    #     51,
    #     54,
    #     55,
    #     56,
    #     57,
    #     59,
    #     60,
    #     61,
    #     62,
    #     63,
    #     65,
    #     66,
    # ]

    ACTIVE_TRIPLETS = [
        2,
        5,
        7,
        10,
        11,
        13,
        14,
        15,
        16,
        20,
        22,
        23,
        31,
        32,
        35,
        38,
        39,
        40,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        51,
        52,
        54,
        57,
        61,
        63,
        65,
    ]

    compact_triplet_to_ivt = train_dataset.get_compact_triplet_to_ivt()

    # Initialize validator
    validator = ModelValidator(
        val_loader=val_loader,
        val_dataset=val_dataset,
        num_classes={
            "instrument": train_dataset.num_classes["instrument"],
            "verb": train_dataset.num_classes["verb"],
            "target": train_dataset.num_classes["target"],
            "triplet": len(ACTIVE_TRIPLETS),
        },
        device=DEVICE,
        model_path=MODEL_PATH,
        triplet_to_ivt=compact_triplet_to_ivt,
        guidance_scale=configs["guidance_scale"],
        hidden_layer_dim=configs["hidden_layer_dim"],
        attention_module_common_dim=configs["attention_module_common_dim"],
        label_mappings=label_mappings,
    )
    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
