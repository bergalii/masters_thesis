import os
import glob
import xml.etree.ElementTree as ET
import statistics
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def visualize_random_samples(annotations_dir, images_dir, class_name_map, seed=None):
    """
    Visualize 2 random images from the dataset with their bounding boxes

    Parameters:
    - annotations_dir: path to XML annotations
    - images_dir: path to image files
    - class_name_map: dictionary mapping class IDs/names
    - seed: random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Get all XML files
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))

    # Select 2 random files
    selected_files = random.sample(xml_files, 2)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    for idx, xml_path in enumerate(selected_files):
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image filename
        image_filename = root.find("filename").text
        image_path = os.path.join(images_dir, image_filename)

        # Read and display image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[idx].imshow(img)
        axs[idx].set_title(f"Image: {image_filename}")

        # Draw bounding boxes
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Create rectangle patch
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axs[idx].add_patch(rect)

            # Add label
            axs[idx].text(
                xmin,
                ymin - 5,
                name,
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        axs[idx].axis("off")

    plt.tight_layout()
    plt.show()


def dataset_summary(annotations_dir, images_dir, class_name_map):
    """
    - annotations_dir: path to the folder containing Pascal VOC .xml files
    - images_dir: path to the folder containing the cropped images
    - class_name_map: dict {class_id -> class_name} or {class_name -> class_name}
    """
    image_info_list = []
    class_distribution = defaultdict(int)
    bboxes_dimensions = []
    actions_per_frame = []
    smallest_bbox = None  # Will store (w, h) of smallest bbox
    largest_bbox = None  # Will store (w, h) of largest bbox

    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_node = root.find("filename")
        if filename_node is None:
            continue
        image_filename = filename_node.text

        size_node = root.find("size")
        if size_node is None:
            continue
        img_width = int(size_node.find("width").text)
        img_height = int(size_node.find("height").text)

        object_nodes = root.findall("object")
        num_actions = len(object_nodes)

        for obj_node in object_nodes:
            name_node = obj_node.find("name")
            if name_node is not None:
                class_label = name_node.text
                class_distribution[class_label] += 1

            bndbox = obj_node.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                w = xmax - xmin
                h = ymax - ymin
                bboxes_dimensions.append((w, h))

                # Update smallest/largest bbox tracking
                area = w * h
                if smallest_bbox is None or area < (
                    smallest_bbox[0] * smallest_bbox[1]
                ):
                    smallest_bbox = (w, h)
                if largest_bbox is None or area > (largest_bbox[0] * largest_bbox[1]):
                    largest_bbox = (w, h)

        actions_per_frame.append(num_actions)

        image_info = {
            "filename": image_filename,
            "width": img_width,
            "height": img_height,
            "num_actions": num_actions,
        }
        image_info_list.append(image_info)

    num_images = len(image_info_list)
    reference_resolution = None
    if num_images > 0:
        reference_resolution = (
            image_info_list[0]["width"],
            image_info_list[0]["height"],
        )
        for info in image_info_list[1:]:
            if (info["width"], info["height"]) != reference_resolution:
                print(
                    f"Warning: {info['filename']} has resolution "
                    f"({info['width']},{info['height']}) != {reference_resolution}."
                )

    median_actions_per_frame = None
    if actions_per_frame:
        median_actions_per_frame = statistics.median(actions_per_frame)

    max_actions = 0
    frame_with_max_actions = None
    for info in image_info_list:
        if info["num_actions"] > max_actions:
            max_actions = info["num_actions"]
            frame_with_max_actions = info["filename"]

    median_bbox_w = None
    median_bbox_h = None
    if bboxes_dimensions:
        widths = [dim[0] for dim in bboxes_dimensions]
        heights = [dim[1] for dim in bboxes_dimensions]
        median_bbox_w = statistics.median(widths)
        median_bbox_h = statistics.median(heights)

    print("--- DATASET SUMMARY ---")
    print(f"Total images: {num_images}")
    if reference_resolution is not None:
        print(f"Image resolution (reference): {reference_resolution}")
    print("\nClass Distribution:")
    for class_name, count in sorted(class_distribution.items()):
        print(f"  {class_name}: {count}")

    print(
        f"\nMedian # of actions per frame: {median_actions_per_frame if median_actions_per_frame else 0}"
    )
    print(f"Frame with max actions: {frame_with_max_actions} (actions = {max_actions})")
    print(f"Median bbox width: {median_bbox_w}, height: {median_bbox_h}")
    print(f"Smallest bbox dimensions: {smallest_bbox[0]} x {smallest_bbox[1]} pixels")
    print(f"Largest bbox dimensions: {largest_bbox[0]} x {largest_bbox[1]} pixels")
    print("----------------------\n")

    return {
        "num_images": num_images,
        "reference_resolution": reference_resolution,
        "class_distribution": dict(class_distribution),
        "median_actions_per_frame": median_actions_per_frame,
        "frame_with_max_actions": frame_with_max_actions,
        "max_actions": max_actions,
        "median_bbox_w": median_bbox_w,
        "median_bbox_h": median_bbox_h,
        "smallest_bbox": smallest_bbox,
        "largest_bbox": largest_bbox,
    }


if __name__ == "__main__":
    annotations_dir = r"dataset\annotations"
    images_dir = r"dataset\images"

    class_name_map = {
        0: "CuttingMesocolon",
        1: "PullingVasDeferens",
        2: "ClippingVasDeferens",
        3: "CuttingVasDeferens",
        4: "ClippingTissue",
        5: "PullingSeminalVesicle",
        6: "ClippingSeminalVesicle",
        7: "CuttingSeminalVesicle",
        8: "SuckingBlood",
        9: "SuckingSmoke",
        10: "PullingTissue",
        11: "CuttingTissue",
        12: "BaggingProstate",
        13: "BladderNeckDissection",
        14: "BladderAnastomosis",
        15: "PullingProstate",
        16: "ClippingBladderNeck",
        17: "CuttingThread",
        18: "UrethraDissection",
        19: "CuttingProstate",
        20: "PullingBladderNeck",
    }

    # 1) Summarize the dataset
    stats = dataset_summary(annotations_dir, images_dir, class_name_map)

    # 2) Visualize 2 random images with bounding boxes
    visualize_random_samples(annotations_dir, images_dir, class_name_map, seed=None)
