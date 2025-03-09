import cv2
import numpy as np
import os
from decord import VideoReader, cpu
from pathlib import Path
import matplotlib

# Force matplotlib to not use any Xwindows backend (important for headless environments)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_circle_borders(frame: np.ndarray, threshold: int = 20):
    """
    Detect the left and right borders of the circular content in a frame.

    Args:
        frame: Input frame as numpy array
        threshold: Brightness threshold for detecting non-black pixels

    Returns:
        Tuple of (left_border, right_border) pixel positions and column sums
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

    return left_border, right_border, col_sums


def visualize_per_frame_borders(
    video_path,
    output_dir="per_frame_visualization",
    target_width=480,
    target_height=270,
    threshold=20,
    margin=5,
    num_frames=5,
):
    """
    Create and save visualizations showing how border detection changes across frames.

    Args:
        video_path: Path to the input video
        output_dir: Directory to save visualizations
        target_width: Width for resized frames
        target_height: Height for resized frames
        threshold: Threshold for border detection
        margin: Margin to add around detected content
        num_frames: Number of frames to sample and visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    video = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(video)

    if total_frames == 0:
        print(f"Error: No frames found in {video_path}")
        return

    print(f"Video loaded: {video_path}")
    print(f"Total frames: {total_frames}")

    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Get all frames
    frames = video.get_batch(frame_indices).asnumpy()

    # Create a figure to show border detection across frames
    fig, axs = plt.subplots(num_frames, 3, figsize=(15, 5 * num_frames))

    # Ensure axs is 2D even with one frame
    if num_frames == 1:
        axs = axs.reshape(1, -1)

    # Create a table to store border information
    border_info = []

    # Process each frame
    for i, (idx, frame) in enumerate(zip(frame_indices, frames)):
        # Detect borders for this specific frame
        left_border, right_border, col_sums = detect_circle_borders(frame, threshold)

        # Apply margin
        left_with_margin = max(0, left_border - margin)
        right_with_margin = min(frame.shape[1] - 1, right_border + margin)

        # Crop and resize the frame
        cropped_frame = frame[:, left_with_margin : right_with_margin + 1, :]
        resized_frame = cv2.resize(
            cropped_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

        # Store border information
        border_info.append(
            {
                "frame": idx,
                "left": left_border,
                "right": right_border,
                "width": right_border - left_border + 1,
                "left_with_margin": left_with_margin,
                "right_with_margin": right_with_margin,
                "width_with_margin": right_with_margin - left_with_margin + 1,
            }
        )

        # Original with detected borders
        axs[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axs[i, 0].axvline(x=left_border, color="r", linestyle="-", linewidth=1)
        axs[i, 0].axvline(x=right_border, color="r", linestyle="-", linewidth=1)
        axs[i, 0].axvline(x=left_with_margin, color="g", linestyle="--", linewidth=1)
        axs[i, 0].axvline(x=right_with_margin, color="g", linestyle="--", linewidth=1)
        axs[i, 0].set_title(f"Frame {idx} Border Detection")
        axs[i, 0].set_ylabel(f"Frame {idx}")

        # Cropped frame
        axs[i, 1].imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title(f"Cropped (width={cropped_frame.shape[1]}px)")

        # Resized frame
        axs[i, 2].imshow(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        axs[i, 2].set_title(f"Resized ({target_width}×{target_height})")

        # Save individual frames
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{idx}_original.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{idx}_cropped.jpg"),
            cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{idx}_resized.jpg"),
            cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR),
        )

    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_frame_borders.png"), dpi=150)
    plt.close()

    # Create a visualization showing how borders change over time
    fig, ax = plt.subplots(figsize=(12, 6))

    frame_nums = [info["frame"] for info in border_info]
    left_borders = [info["left"] for info in border_info]
    right_borders = [info["right"] for info in border_info]
    content_widths = [info["width"] for info in border_info]

    ax.plot(frame_nums, left_borders, "r-", label="Left Border")
    ax.plot(frame_nums, right_borders, "b-", label="Right Border")
    ax.plot(frame_nums, content_widths, "g-", label="Content Width")

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Pixel Position / Width")
    ax.set_title("Border Positions Across Frames")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "border_changes.png"), dpi=150)
    plt.close()

    # Print border information
    print("\nBorder detection information:")
    print("-" * 80)
    print(
        f"{'Frame':<10} {'Left':<10} {'Right':<10} {'Width':<10} {'Left+Margin':<15} {'Right+Margin':<15} {'Width+Margin':<15}"
    )
    print("-" * 80)

    for info in border_info:
        print(
            f"{info['frame']:<10} {info['left']:<10} {info['right']:<10} {info['width']:<10} {info['left_with_margin']:<15} {info['right_with_margin']:<15} {info['width_with_margin']:<15}"
        )

    print("-" * 80)

    # Calculate statistics
    left_min = min(left_borders)
    left_max = max(left_borders)
    right_min = min(right_borders)
    right_max = max(right_borders)
    width_min = min(content_widths)
    width_max = max(content_widths)

    print(f"\nBorder position range:")
    print(f"Left border: {left_min} to {left_max} (variation: {left_max - left_min}px)")
    print(
        f"Right border: {right_min} to {right_max} (variation: {right_max - right_min}px)"
    )
    print(
        f"Content width: {width_min} to {width_max} (variation: {width_max - width_min}px)"
    )

    print(f"\nAll visualizations saved to {output_dir}")


def main():
    # Directory path
    CLIPS_DIR = r"05_datasets_dir/CholecT50/videos"
    OUTPUT_DIR = r"per_frame_visualization"

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

    # Visualize per-frame border detection
    visualize_per_frame_borders(
        first_video,
        output_dir=OUTPUT_DIR,
        target_width=480,
        target_height=270,
        threshold=20,
        margin=5,
        num_frames=10,  # Increase number of frames to better see changes
    )


if __name__ == "__main__":
    main()
