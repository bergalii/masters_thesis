import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 1) Updated Class Name Map
#    Assign these IDs in YOLO accordingly (0, 1, 2, ...), or adapt based on your annotation IDs.
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


def get_image_txt_pairs(folder_path):
    """
    Returns dictionaries for .jpg and .txt keyed by filename (stem).
    """
    jpg_files = {}
    txt_files = {}

    for file_name in os.listdir(folder_path):
        lower_name = file_name.lower()
        if lower_name.endswith(".jpg"):
            stem = os.path.splitext(file_name)[0]
            jpg_files[stem] = file_name
        elif lower_name.endswith(".txt"):
            stem = os.path.splitext(file_name)[0]
            txt_files[stem] = file_name

    return jpg_files, txt_files


def remove_unmatched_files(folder_path):
    """
    1) Remove images with no corresponding text file.
    2) Remove empty text files and their corresponding images.
    Returns a list of valid file stems that survived.
    """
    jpg_files, txt_files = get_image_txt_pairs(folder_path)
    valid_stems = []

    # 1. Remove images with no matching .txt
    for stem, jpg_name in list(jpg_files.items()):
        if stem not in txt_files:
            jpg_path = os.path.join(folder_path, jpg_name)
            os.remove(jpg_path)
            print(f"Deleted unmatched image: {jpg_name}")

    # Refresh after deletion
    jpg_files, txt_files = get_image_txt_pairs(folder_path)

    # 2. Remove empty .txt files and their .jpg
    for stem, txt_name in list(txt_files.items()):
        txt_path = os.path.join(folder_path, txt_name)
        if os.path.getsize(txt_path) == 0:
            # Remove empty text file
            os.remove(txt_path)
            print(f"Deleted empty text file: {txt_name}")
            # Remove corresponding image
            if stem in jpg_files:
                jpg_path = os.path.join(folder_path, jpg_files[stem])
                os.remove(jpg_path)
                print(f"Deleted image corresponding to empty text: {jpg_files[stem]}")
        else:
            # If text file not empty and has matching .jpg, consider it valid
            if stem in jpg_files:
                valid_stems.append(stem)

    return valid_stems


def parse_yolo_annotations(txt_path, img_width, img_height):
    """
    Parse YOLO annotations and return bounding boxes in absolute coords.
    Each bbox is a dict: {
      'class_id': int,
      'xmin': float,
      'ymin': float,
      'xmax': float,
      'ymax': float
    }
    Also print filename if malformed lines occur.
    """
    bboxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                # 4) Print the filename if malformed lines occur
                print(f"Malformed line in {txt_path}: {line}")
                continue  # skip malformed lines

            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height

            xmin = x_center - (w / 2.0)
            ymin = y_center - (h / 2.0)
            xmax = x_center + (w / 2.0)
            ymax = y_center + (h / 2.0)

            bboxes.append(
                {
                    "class_id": class_id,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )
    return bboxes


def detect_black_bars_and_crop(img, filename, threshold=20):
    """
    Automatically detect black bars by scanning columns from the left and right
    until pixel values exceed a given threshold.
    If the detected range is invalid or doesn't match expectations, print a warning.

    :param img: The original image (BGR).
    :param filename: Name of the file (for printing warnings).
    :param threshold: Mean pixel intensity threshold to consider a column "black".
    :return: (cropped_img, left_crop, right_crop)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Find left boundary
    left_crop = 0
    while left_crop < width:
        col_mean = gray[:, left_crop].mean()
        if col_mean > threshold:
            break
        left_crop += 1

    # Find right boundary
    right_crop = width - 1
    while right_crop >= 0:
        col_mean = gray[:, right_crop].mean()
        if col_mean > threshold:
            break
        right_crop -= 1

    # Validate the detected range
    if left_crop >= right_crop:
        # This means the "black bar" logic failed.
        print(
            f"Warning: {filename} -> invalid black-bar crop range: left={left_crop}, right={right_crop}"
        )
        # Return the original image unmodified, with no offset
        return img, 0, width - 1

    cropped_img = img[:, left_crop : right_crop + 1]
    return cropped_img, left_crop, right_crop


def adjust_bboxes_for_crop(bboxes, left_crop):
    """
    Shift bounding boxes horizontally after cropping 'left_crop' columns.
    (No vertical crop in this example.)
    """
    adjusted = []
    for box in bboxes:
        adjusted.append(
            {
                "class_id": box["class_id"],
                "xmin": box["xmin"] - left_crop,
                "ymin": box["ymin"],
                "xmax": box["xmax"] - left_crop,
                "ymax": box["ymax"],
            }
        )
    return adjusted


def create_pascal_voc_xml(filename, width, height, bboxes, class_name_map, save_path):
    """
    Create a Pascal VOC style XML file with required modifications:
      - Remove 'folder', 'depth', 'segmented', 'truncated', 'occluded', 'difficult', 'rotation'.
    """
    # Root <annotation>
    annotation = ET.Element("annotation")

    # 2) Remove 'folder', 'depth', 'segmented'
    #    So we do NOT create them at all.

    # <filename>
    filename_el = ET.SubElement(annotation, "filename")
    filename_el.text = filename

    # <size>
    size_el = ET.SubElement(annotation, "size")
    width_el = ET.SubElement(size_el, "width")
    width_el.text = str(width)
    height_el = ET.SubElement(size_el, "height")
    height_el.text = str(height)
    # We omit <depth> tag

    # We omit <segmented>

    # For each bbox, create <object>
    for box in bboxes:
        class_id = box["class_id"]
        class_name = class_name_map.get(class_id, f"class_{class_id}")
        obj_el = ET.SubElement(annotation, "object")

        name_el = ET.SubElement(obj_el, "name")
        name_el.text = class_name

        # We omit 'truncated', 'occluded', 'difficult'

        bndbox_el = ET.SubElement(obj_el, "bndbox")
        xmin_el = ET.SubElement(bndbox_el, "xmin")
        xmin_el.text = str(int(round(box["xmin"])))
        ymin_el = ET.SubElement(bndbox_el, "ymin")
        ymin_el.text = str(int(round(box["ymin"])))
        xmax_el = ET.SubElement(bndbox_el, "xmax")
        xmax_el.text = str(int(round(box["xmax"])))
        ymax_el = ET.SubElement(bndbox_el, "ymax")
        ymax_el.text = str(int(round(box["ymax"])))

        # We omit the 'rotation' attributes or any <attributes> block

    # Pretty-print
    xml_str = ET.tostring(annotation, "utf-8")
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml_str = parsed_xml.toprettyxml(indent="  ")

    with open(save_path, "w") as f:
        f.write(pretty_xml_str)


def main(folder_path, output_folder):
    """
    Main pipeline:
      1. Remove unmatched/empty label files.
      2. For each valid pair, parse YOLO bboxes.
      3. Crop fixed black bars from left/right.
      4. Adjust bboxes.
      5. Save cropped image -> output_folder/images
      6. Save Pascal VOC XML -> output_folder/annotations
    """

    # Make output subdirectories
    images_out = os.path.join(output_folder, "images")
    annots_out = os.path.join(output_folder, "annotations")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(annots_out, exist_ok=True)

    # Step 1: Remove unmatched / empty
    valid_stems = remove_unmatched_files(folder_path)

    for stem in valid_stems:
        img_path = os.path.join(folder_path, stem + ".jpg")
        txt_path = os.path.join(folder_path, stem + ".txt")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        h, w, _ = img.shape

        # Step 2: Parse YOLO bboxes
        yolo_bboxes = parse_yolo_annotations(txt_path, w, h)
        # If no bboxes, you might want to skip or handle differently
        if not yolo_bboxes:
            print(f"No bounding boxes found in {txt_path}, skipping {stem}.")
            continue

        # Step 3: Crop black bars (fixed positions)
        cropped_img, left_offset, right_offset = detect_black_bars_and_crop(img, stem)
        new_h, new_w = cropped_img.shape[:2]  # after cropping

        # Step 4: Adjust bounding boxes
        adjusted_bboxes = adjust_bboxes_for_crop(yolo_bboxes, left_offset)

        # Step 5: Save cropped image in images folder
        new_image_name = f"{stem}_cropped.jpg"
        new_image_path = os.path.join(images_out, new_image_name)
        cv2.imwrite(new_image_path, cropped_img)

        # Step 6: Create Pascal VOC XML in annotations folder
        xml_name = f"{stem}.xml"
        xml_save_path = os.path.join(annots_out, xml_name)

        create_pascal_voc_xml(
            filename=new_image_name,  # <filename> in XML
            width=new_w,
            height=new_h,
            bboxes=adjusted_bboxes,
            class_name_map=class_name_map,
            save_path=xml_save_path,
        )

        # print(f"Processed {stem} -> {new_image_name} + {xml_name}")


if __name__ == "__main__":

    folder_path = r"C:\Users\Berk Cam\Downloads\val\val\obj"
    output_folder = "dataset"

    main(folder_path, output_folder)
