import pandas as pd
from decord import VideoReader, cpu


def check_corrupted_videos(clips_dir, annotations_path):
    """Check videos using the same method as your dataset"""

    # Read annotations like your dataset does
    annotations = pd.read_csv(annotations_path)

    corrupted = []
    valid = []

    print(f"Checking {len(annotations)} videos from annotations...")

    for i, row in annotations.iterrows():
        file_name = row["file_name"]
        video_path = f"{clips_dir}/{file_name}"

        print(f"{i+1}/{len(annotations)}: {file_name}", end=" ... ")

        try:
            # Same as your dataset code
            original_video = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(original_video)

            if total_frames > 0:
                print("OK")
                valid.append(file_name)
            else:
                print("CORRUPTED (no frames)")
                corrupted.append(file_name)

        except Exception as e:
            print("CORRUPTED")
            corrupted.append(file_name)

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"Valid videos: {len(valid)}")
    print(f"Corrupted videos: {len(corrupted)}")
    print(f"{'='*50}")

    if corrupted:
        print("\nCORRUPTED VIDEOS:")
        for video in corrupted:
            print(f"  - {video}")

        # Save corrupted list to file
        with open("corrupted_videos.txt", "w") as f:
            for video in corrupted:
                f.write(f"{video}\n")
        print(f"\nCorrupted videos saved to: corrupted_videos.txt")

    return corrupted, valid


if __name__ == "__main__":
    # Same paths as your training script
    CLIPS_DIR = "/data/Berk/masters_thesis/05_datasets_dir/UKE/clips"
    ANNOTATIONS_PATH = "/data/Berk/masters_thesis/05_datasets_dir/UKE/gt.csv"

    corrupted, valid = check_corrupted_videos(CLIPS_DIR, ANNOTATIONS_PATH)
