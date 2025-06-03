import cv2
import numpy as np
import os
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
import random


def detect_circle_borders(frame: np.ndarray, threshold: int = 20):
    """
    Detect the left and right borders of the circular content in a frame.

    Args:
        frame: Input frame as numpy array (RGB format)
        threshold: Brightness threshold for detecting non-black pixels

    Returns:
        Tuple of (left_border, right_border) pixel positions
    """
    # Sum pixel values across columns (vertical strips)
    col_sums = np.sum(frame, axis=(0, 2))
    col_mask = col_sums > threshold * frame.shape[0]  # Scale threshold by height

    # Find where content begins (left border) and ends (right border)
    non_zero_indices = np.where(col_mask)[0]

    if len(non_zero_indices) > 0:
        left_border = non_zero_indices[0]
        right_border = non_zero_indices[-1]
    else:
        # Fallback if detection fails
        left_border = 0
        right_border = frame.shape[1] - 1

    return left_border, right_border


def visualize_border_detection(video_path, output_path, threshold=20, margin=5):
    """
    Create and save a simple visualization showing one original frame and its cropped version.

    Args:
        video_path: Path to the input video
        output_path: Path to save the visualization image
        threshold: Threshold for border detection
        margin: Margin to add around detected content
    """
    # Load video
    video = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(video)

    if total_frames == 0:
        print(f"Error: No frames found in {video_path}")
        return

    # Select a frame from the middle of the video
    frame_idx = total_frames // 2

    # Get the frame
    frame = video[frame_idx].asnumpy()

    # Detect borders
    left_border, right_border = detect_circle_borders(frame, threshold)

    # Apply margin
    left_with_margin = max(0, left_border - margin)
    right_with_margin = min(frame.shape[1] - 1, right_border + margin)

    # Crop the frame
    cropped_frame = frame[:, left_with_margin : right_with_margin + 1, :]

    # Create a visualization with frames stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Original with detected borders
    axs[0].imshow(frame)
    axs[0].axvline(x=left_border, color="r", linestyle="-", linewidth=2)
    axs[0].axvline(x=right_border, color="r", linestyle="-", linewidth=2)
    axs[0].set_title(f"Original Frame with Detected Borders")
    axs[0].axis("off")

    # Cropped frame
    axs[1].imshow(cropped_frame)
    axs[1].set_title(f"Cropped Frame")
    axs[1].axis("off")

    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Visualization saved to {output_path}")


def main():
    # Directory path
    CLIPS_DIR = r"05_datasets_dir/UKE/clips"
    OUTPUT_PATH = r"border_detection.png"

    # Get the first video file from the directory
    video_files = []
    for file in os.listdir(CLIPS_DIR):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_files.append(file)

    if not video_files:
        print(f"No video files found in {CLIPS_DIR}")
        return

    # Sort and get the first video
    video_files.sort()
    first_video = os.path.join(CLIPS_DIR, video_files[0])
    print(f"Analyzing video: {first_video}")

    # Create the simple visualization
    visualize_border_detection(
        first_video, output_path=OUTPUT_PATH, threshold=20, margin=5
    )


if __name__ == "__main__":
    main()
