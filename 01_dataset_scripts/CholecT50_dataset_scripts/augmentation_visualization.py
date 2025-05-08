import matplotlib.pyplot as plt
import numpy as np
import cv2
from decord import VideoReader, cpu
import ast


def visualize_augmentations(
    dataset, idx, num_frames=4, save_path="augmentation_example.png"
):
    """
    Visualize original and augmented frames from the dataset.

    Args:
        dataset: MultiTaskVideoDataset instance
        idx: Index of the sample to visualize
        num_frames: Number of frames to visualize (default: 4)
        save_path: Path to save the visualization (default: "augmentation_example.png")

    Returns:
        None (saves the visualization to the specified path)
    """
    # Get the original video path
    row = dataset.annotations.iloc[idx]
    video_path = f"{dataset.clips_dir}/{row['file_name']}"

    # Load the video
    original_video = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(original_video)

    # Get evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Extract the original frames
    original_frames = original_video.get_batch(indices).asnumpy()

    # Simply resize the frames without cropping black borders
    processed_original_frames = []
    for frame in original_frames:
        # Resize to the target dimensions
        resized_frame = cv2.resize(
            frame,
            (dataset.frame_width, dataset.frame_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        processed_original_frames.append(resized_frame)

    # Apply augmentations
    augmented_frames = []
    for orig_frame in processed_original_frames:
        # Apply the same transform to each frame
        data = dataset.transform(image=orig_frame.copy())
        augmented_frames.append(data["image"])

    # Don't convert the color space
    orig_frames_display = processed_original_frames
    aug_frames_display = augmented_frames

    # Create the visualization grid
    fig, axes = plt.subplots(num_frames, 2, figsize=(10, 3 * num_frames))

    for i in range(num_frames):
        # Original frame
        axes[i, 0].imshow(orig_frames_display[i])
        axes[i, 0].set_title(f"Original Frame {i+1}")
        axes[i, 0].axis("off")

        # Augmented frame
        axes[i, 1].imshow(aug_frames_display[i])
        axes[i, 1].set_title(f"Augmented Frame {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.close()

    # Print information about the sample
    print(f"Video ID: {row['video_id']}")
    print(f"File: {row['file_name']}")

    # Print the labeled tasks
    print("\nLabels:")
    print(
        f"Instrument: {[dataset.label_mappings['instrument'][i] for i in ast.literal_eval(str(row['instrument_label']))]}"
    )
    print(
        f"Verb: {[dataset.label_mappings['verb'][i] for i in ast.literal_eval(str(row['verb_label']))]}"
    )
    print(
        f"Target: {[dataset.label_mappings['target'][i] for i in ast.literal_eval(str(row['target_label']))]}"
    )

    # Print the augmentations applied
    print("\nAugmentations applied:")
    for i, transform in enumerate(dataset.transform.transforms):
        print(f"{i+1}. {transform.__class__.__name__}")
