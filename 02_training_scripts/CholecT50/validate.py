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
        attention_module_common_dim: int,
        hidden_layer_dim: int,
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
        """Validate the model and compute metrics"""
        self.model.eval()

        # Initialize recognition object
        recognize = Recognition(num_class=self.num_classes["triplet"])
        # recognize.reset_global()
        recognize.reset()

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

                    # Update with triplet predictions and the label for the video
                    predictions = guided_triplet_probs.cpu().numpy()
                    labels = batch_labels["triplet"][b].unsqueeze(0).cpu().numpy()

                    # Update the recognizer with the current video
                    recognize.update(labels, predictions)

                    # Signal end of video to the Recognition evaluator
                    # recognize.video_end()
        # Compute video-level AP metrics
        # results = {
        #     "triplet": recognize.compute_video_AP("ivt"),
        #     "verb": recognize.compute_video_AP("v"),
        #     "instrument": recognize.compute_video_AP("i"),
        #     "target": recognize.compute_video_AP("t"),
        # }

        # print("\nVideo-level Validation Results:")
        # for task in ["verb", "instrument", "target", "triplet"]:
        #     print(f"{task.capitalize()} Results:")
        #     print(f"Video-level mAP: {results[task]['mAP']:.4f}")
        #     print(f"Video-level AP: {results[task]['AP']}")

        global_results = {
            "triplet": recognize.compute_AP("ivt"),
            "verb": recognize.compute_AP("v"),
            "instrument": recognize.compute_AP("i"),
            "target": recognize.compute_AP("t"),
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
    MODEL_PATH = r"04_models_dir/training_20250428_094504/best_model_student.pth"

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
        hidden_layer_dim=configs["hidden_layer_dim"],
        attention_module_common_dim=configs["attention_module_common_dim"],
    )
    # Run validation
    validator.validate()


if __name__ == "__main__":
    main()
