import torch
from torch import nn
from torchvision.models.video.swin_transformer import swin3d_s, Swin3D_S_Weights
from recognition import Recognition
from dataset import MultiTaskVideoDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import set_seeds, load_configs
from modules import MultiTaskHead, AttentionModule
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SemanticAnalyzer:
    def __init__(self):
        # Store results for teacher and student
        self.teacher_semantic_similarities = []
        self.student_semantic_similarities = []
        self.random_similarities = []
        self.teacher_examples = []
        self.student_examples = []

    def count_shared_components(
        self, pred1_indices, pred2_indices, triplet_to_ivt, index_to_triplet
    ):
        """Count shared components between two prediction sets"""
        shared_components = []

        for p1_idx in pred1_indices:
            for p2_idx in pred2_indices:
                if p1_idx == p2_idx:
                    continue

                # Get original triplet IDs
                try:
                    triplet1_id = index_to_triplet[p1_idx]
                    triplet2_id = index_to_triplet[p2_idx]

                    if triplet1_id in triplet_to_ivt and triplet2_id in triplet_to_ivt:
                        components1 = triplet_to_ivt[triplet1_id]
                        components2 = triplet_to_ivt[triplet2_id]

                        # Count shared components (instrument, verb, target)
                        shared = sum(
                            1 for i in range(3) if components1[i] == components2[i]
                        )
                        shared_components.append(shared)
                except:
                    continue

        return np.mean(shared_components) if shared_components else 0

    def analyze_sample(
        self,
        teacher_probs,
        student_probs,
        gt_indices,
        triplet_to_ivt,
        index_to_triplet,
        top_k=5,
    ):
        """Analyze semantic relationships for teacher and student predictions"""

        # Get top-k predictions for teacher and student
        teacher_top_indices = np.argsort(teacher_probs)[-top_k:]
        student_top_indices = np.argsort(student_probs)[-top_k:]

        # Generate random baseline
        all_indices = np.arange(len(teacher_probs))
        random_indices = np.random.choice(all_indices, size=top_k, replace=False)

        # Calculate semantic similarities
        teacher_semantic = self.count_shared_components(
            gt_indices, teacher_top_indices, triplet_to_ivt, index_to_triplet
        )
        student_semantic = self.count_shared_components(
            gt_indices, student_top_indices, triplet_to_ivt, index_to_triplet
        )
        random_semantic = self.count_shared_components(
            gt_indices, random_indices, triplet_to_ivt, index_to_triplet
        )

        # Store example for detailed analysis
        if len(self.teacher_examples) < 5:  # Store first 5 examples
            example = {
                "gt_indices": gt_indices,
                "teacher_top": teacher_top_indices,
                "student_top": student_top_indices,
                "teacher_probs": teacher_probs[teacher_top_indices],
                "student_probs": student_probs[student_top_indices],
                "teacher_semantic": teacher_semantic,
                "student_semantic": student_semantic,
            }
            self.teacher_examples.append(example)

        return teacher_semantic, student_semantic, random_semantic

    def add_sample_analysis(
        self,
        teacher_probs,
        student_probs,
        gt_indices,
        triplet_to_ivt,
        index_to_triplet,
        top_k=5,
    ):
        """Add analysis for a single sample"""
        teacher_sem, student_sem, random_sem = self.analyze_sample(
            teacher_probs,
            student_probs,
            gt_indices,
            triplet_to_ivt,
            index_to_triplet,
            top_k,
        )

        self.teacher_semantic_similarities.append(teacher_sem)
        self.student_semantic_similarities.append(student_sem)
        self.random_similarities.append(random_sem)

    def generate_report(self):
        """Generate final report"""
        teacher_array = np.array(self.teacher_semantic_similarities)
        student_array = np.array(self.student_semantic_similarities)
        random_array = np.array(self.random_similarities)

        # Basic statistics
        teacher_mean = np.mean(teacher_array)
        teacher_std = np.std(teacher_array)
        student_mean = np.mean(student_array)
        student_std = np.std(student_array)
        random_mean = np.mean(random_array)
        random_std = np.std(random_array)

        # Statistical tests
        t_stat_teacher, p_value_teacher = stats.ttest_rel(teacher_array, random_array)
        t_stat_student, p_value_student = stats.ttest_rel(student_array, random_array)
        t_stat_comparison, p_value_comparison = stats.ttest_rel(
            teacher_array, student_array
        )

        print("\n" + "=" * 80)
        print("SELF-DISTILLATION SEMANTIC ANALYSIS REPORT")
        print("=" * 80)

        print(f"\nSample Statistics:")
        print(f"  Total samples analyzed: {len(teacher_array)}")

        print(f"\nSemantic Similarity Results:")
        print(f"  Teacher Model: {teacher_mean:.4f} ± {teacher_std:.4f}")
        print(f"  Student Model: {student_mean:.4f} ± {student_std:.4f}")
        print(f"  Random Baseline: {random_mean:.4f} ± {random_std:.4f}")

        print(f"\nStatistical Significance:")
        print(f"  Teacher vs Random: p = {p_value_teacher:.4f}")
        print(f"  Student vs Random: p = {p_value_student:.4f}")
        print(f"  Teacher vs Student: p = {p_value_comparison:.4f}")

        # Improvements
        teacher_improvement = (
            ((teacher_mean - random_mean) / random_mean) * 100 if random_mean > 0 else 0
        )
        student_improvement = (
            ((student_mean - random_mean) / random_mean) * 100 if random_mean > 0 else 0
        )

        print(f"\nImprovement over Random:")
        print(f"  Teacher: {teacher_improvement:.1f}%")
        print(f"  Student: {student_improvement:.1f}%")

        # Knowledge transfer
        if teacher_mean > 0:
            transfer_ratio = (student_mean / teacher_mean) * 100
            print(f"\nKnowledge Transfer:")
            print(
                f"  Student retained {transfer_ratio:.1f}% of teacher's semantic knowledge"
            )

        # Conclusion
        print(f"\nConclusion:")
        if p_value_teacher < 0.05 and teacher_mean > random_mean:
            print("  ✓ Teacher learned semantic relationships")
        else:
            print("  ✗ Teacher did not learn significant semantic relationships")

        if p_value_student < 0.05 and student_mean > random_mean:
            print("  ✓ Student benefited from semantic knowledge transfer")
        else:
            print("  ✗ Student did not benefit from semantic knowledge transfer")

        return {
            "teacher_mean": teacher_mean,
            "student_mean": student_mean,
            "random_mean": random_mean,
            "p_teacher": p_value_teacher,
            "p_student": p_value_student,
            "teacher_improvement": teacher_improvement,
            "student_improvement": student_improvement,
        }


class ModelValidator:
    def __init__(
        self,
        val_loader,
        val_dataset,
        num_classes: dict,
        device: str,
        teacher_model_path: str,
        student_model_path: str,
        triplet_to_ivt: dict,
        attention_module_common_dim: int,
        hidden_layer_dim: int,
        guidance_scale: float,
    ):
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = device
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
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
        self.label_mappings = val_dataset.label_mappings

        # Initialize models
        self.teacher_model = self._initialize_model(teacher_model_path)
        self.student_model = self._initialize_model(student_model_path)

    def _initialize_model(self, model_path):
        """Initialize the model architecture and load the saved weights"""
        model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT).to(self.device)
        model.head = nn.Identity()
        in_features = model.num_features

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
            in_features
            + self.attention_module_common_dim
            + sum(self.feature_dims.values())
        )

        model.triplet_head = MultiTaskHead(
            total_input_size, self.num_classes["triplet"], self.hidden_layer_dim
        ).to(self.device)

        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def _forward_pass(self, model, inputs):
        """Your original forward pass - unchanged"""
        backbone_features = model(inputs)
        verb_logits, verb_hidden = model.verb_head(backbone_features)
        instrument_logits, inst_hidden = model.instrument_head(backbone_features)
        target_logits, target_hidden = model.target_head(backbone_features)

        attention_output = model.attention_module(
            verb_hidden,
            verb_logits,
            inst_hidden,
            instrument_logits,
            target_hidden,
            target_logits,
        )

        combined_features = torch.cat(
            [
                backbone_features,
                attention_output,
                torch.sigmoid(verb_logits),
                torch.sigmoid(instrument_logits),
                torch.sigmoid(target_logits),
            ],
            dim=1,
        )

        triplet_logits, _ = model.triplet_head(combined_features)

        return {
            "verb": verb_logits,
            "instrument": instrument_logits,
            "target": target_logits,
            "triplet": triplet_logits,
        }

    def run_semantic_analysis(self):
        """Run semantic analysis using your original validation loop"""
        self.teacher_model.eval()
        self.student_model.eval()

        semantic_analyzer = SemanticAnalyzer()

        with torch.no_grad():
            for inputs_batch, batch_labels in self.val_loader:
                batch_size = inputs_batch.shape[0]
                num_clips = inputs_batch.shape[1]

                for b in range(batch_size):
                    # Your original video processing logic - unchanged
                    teacher_video_outputs = {
                        task: [] for task in ["verb", "instrument", "target", "triplet"]
                    }
                    student_video_outputs = {
                        task: [] for task in ["verb", "instrument", "target", "triplet"]
                    }

                    for c in range(num_clips):
                        clip = inputs_batch[b, c].unsqueeze(0).to(self.device)

                        # Get predictions from both models
                        teacher_outputs = self._forward_pass(self.teacher_model, clip)
                        student_outputs = self._forward_pass(self.student_model, clip)

                        for task, outputs in teacher_outputs.items():
                            teacher_video_outputs[task].append(outputs)
                        for task, outputs in student_outputs.items():
                            student_video_outputs[task].append(outputs)

                    # Your original averaging logic - unchanged
                    teacher_task_logits = {}
                    student_task_logits = {}

                    for task, outputs_list in teacher_video_outputs.items():
                        outputs_tensor = torch.cat([o for o in outputs_list], dim=0)
                        teacher_task_logits[task] = torch.mean(
                            outputs_tensor, dim=0, keepdim=True
                        )

                    for task, outputs_list in student_video_outputs.items():
                        outputs_tensor = torch.cat([o for o in outputs_list], dim=0)
                        student_task_logits[task] = torch.mean(
                            outputs_tensor, dim=0, keepdim=True
                        )

                    # Convert to probabilities
                    teacher_task_probabilities = {
                        task: torch.sigmoid(logits)
                        for task, logits in teacher_task_logits.items()
                    }
                    student_task_probabilities = {
                        task: torch.sigmoid(logits)
                        for task, logits in student_task_logits.items()
                    }

                    # Get final triplet probabilities (your original guidance logic)
                    teacher_guidance_inst = torch.matmul(
                        teacher_task_probabilities["instrument"], self.MI
                    )
                    teacher_guidance_verb = torch.matmul(
                        teacher_task_probabilities["verb"], self.MV
                    )
                    teacher_guidance_target = torch.matmul(
                        teacher_task_probabilities["target"], self.MT
                    )
                    teacher_guidance = (
                        teacher_guidance_inst
                        * teacher_guidance_verb
                        * teacher_guidance_target
                    )
                    teacher_guided_triplet_probs = (
                        1 - self.guidance_scale
                    ) * teacher_task_probabilities["triplet"] + self.guidance_scale * (
                        teacher_guidance * teacher_task_probabilities["triplet"]
                    )

                    student_guidance_inst = torch.matmul(
                        student_task_probabilities["instrument"], self.MI
                    )
                    student_guidance_verb = torch.matmul(
                        student_task_probabilities["verb"], self.MV
                    )
                    student_guidance_target = torch.matmul(
                        student_task_probabilities["target"], self.MT
                    )
                    student_guidance = (
                        student_guidance_inst
                        * student_guidance_verb
                        * student_guidance_target
                    )
                    student_guided_triplet_probs = (
                        1 - self.guidance_scale
                    ) * student_task_probabilities["triplet"] + self.guidance_scale * (
                        student_guidance * student_task_probabilities["triplet"]
                    )

                    # Get ground truth indices
                    gt_triplet_labels = batch_labels["triplet"][b]
                    gt_indices = torch.where(gt_triplet_labels == 1)[0].cpu().numpy()

                    if len(gt_indices) > 0:
                        # Run semantic analysis
                        teacher_probs = teacher_guided_triplet_probs.cpu().numpy()[0]
                        student_probs = student_guided_triplet_probs.cpu().numpy()[0]

                        semantic_analyzer.add_sample_analysis(
                            teacher_probs,
                            student_probs,
                            gt_indices,
                            self.val_dataset.triplet_to_ivt,
                            self.val_dataset.index_to_triplet,
                        )

        # Generate report
        results = semantic_analyzer.generate_report()
        return semantic_analyzer, results


def main():
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    ANNOTATIONS_PATH = r"05_datasets_dir/CholecT50/annotations.csv"
    CONFIGS_PATH = r"02_training_scripts/CholecT50/configs.yaml"
    TEACHER_MODEL_PATH = (
        r"04_models_dir/training_20250505_211505/best_model_teacher.pth"
    )
    STUDENT_MODEL_PATH = (
        r"04_models_dir/training_20250621_111215/best_model_student.pth"
    )

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
        teacher_model_path=TEACHER_MODEL_PATH,
        student_model_path=STUDENT_MODEL_PATH,
        triplet_to_ivt=val_dataset.triplet_to_ivt,
        guidance_scale=configs["guidance_scale"],
        hidden_layer_dim=configs["hidden_layer_dim"],
        attention_module_common_dim=configs["attention_module_common_dim"],
    )

    # Run semantic analysis
    print("Running semantic analysis...")
    analyzer, results = validator.run_semantic_analysis()


if __name__ == "__main__":
    main()
