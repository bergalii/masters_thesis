import matplotlib.pyplot as plt
import re
from collections import defaultdict
import numpy as np


def parse_data_file(filename):
    """Parse the data file and extract information for each validation split."""
    with open(filename, "r") as f:
        content = f.read()

    # Split by validation splits
    splits = re.split(r"val_split: (\d+)", content)[1:]  # Remove empty first element

    data = {}
    for i in range(0, len(splits), 2):
        split_num = int(splits[i])
        split_content = splits[i + 1]

        # Extract instruments
        instruments = {}
        instrument_section = re.search(
            r"INSTRUMENT LABELS:(.*?)VERB LABELS:", split_content, re.DOTALL
        )
        if instrument_section:
            for line in instrument_section.group(1).strip().split("\n"):
                if ":" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        idx = parts[0].strip()
                        name_counts = parts[1].strip().split(" - ")
                        if len(name_counts) >= 2:
                            name = name_counts[0].strip()
                            counts = name_counts[1].split(", ")
                            if len(counts) >= 2:
                                instruments[name] = (int(counts[0]), int(counts[1]))

        # Extract verbs
        verbs = {}
        verb_section = re.search(
            r"VERB LABELS:(.*?)TARGET LABELS:", split_content, re.DOTALL
        )
        if verb_section:
            for line in verb_section.group(1).strip().split("\n"):
                if ":" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        idx = parts[0].strip()
                        name_counts = parts[1].strip().split(" - ")
                        if len(name_counts) >= 2:
                            name = name_counts[0].strip()
                            counts = name_counts[1].split(", ")
                            if len(counts) >= 2:
                                verbs[name] = (int(counts[0]), int(counts[1]))

        # Extract targets
        targets = {}
        target_section = re.search(
            r"TARGET LABELS:(.*?)TRIPLET LABELS:", split_content, re.DOTALL
        )
        if target_section:
            for line in target_section.group(1).strip().split("\n"):
                if ":" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        idx = parts[0].strip()
                        name_counts = parts[1].strip().split(" - ")
                        if len(name_counts) >= 2:
                            name = name_counts[0].strip()
                            counts = name_counts[1].split(", ")
                            if len(counts) >= 2:
                                targets[name] = (int(counts[0]), int(counts[1]))

        # Extract ALL triplets
        triplets = {}
        triplet_section = re.search(r"TRIPLET LABELS:(.*?)$", split_content, re.DOTALL)
        if triplet_section:
            for line in triplet_section.group(1).strip().split("\n"):
                if ":" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        idx = parts[0].strip()
                        name_counts = parts[1].strip().split(" - ")
                        if len(name_counts) >= 2:
                            name = name_counts[0].strip()
                            counts = name_counts[1].split(", ")
                            if len(counts) >= 2:
                                triplets[name] = (int(counts[0]), int(counts[1]))

        data[split_num] = {
            "instruments": instruments,
            "verbs": verbs,
            "targets": targets,
            "triplets": triplets,
        }

    return data


def calculate_averages(data):
    """Calculate average train counts across all validation splits."""
    all_instruments = defaultdict(list)
    all_verbs = defaultdict(list)
    all_targets = defaultdict(list)

    # Collect all train counts for each category
    for split_data in data.values():
        # Instruments
        for name, (train_count, val_count) in split_data["instruments"].items():
            all_instruments[name].append(train_count)

        # Verbs
        for name, (train_count, val_count) in split_data["verbs"].items():
            all_verbs[name].append(train_count)

        # Targets
        for name, (train_count, val_count) in split_data["targets"].items():
            all_targets[name].append(train_count)

    # Calculate averages of train counts
    avg_instruments = {
        name: np.mean(counts) for name, counts in all_instruments.items()
    }
    avg_verbs = {name: np.mean(counts) for name, counts in all_verbs.items()}
    avg_targets = {name: np.mean(counts) for name, counts in all_targets.items()}

    return avg_instruments, avg_verbs, avg_targets


def calculate_triplet_ratios(data):
    """Calculate train/val ratios for triplets across all splits."""
    all_triplets = {}

    # Collect all triplets and their train/val counts
    for split_data in data.values():
        for name, (train_count, val_count) in split_data["triplets"].items():
            if name not in all_triplets:
                all_triplets[name] = {"train": [], "val": []}
            all_triplets[name]["train"].append(train_count)
            all_triplets[name]["val"].append(val_count)

    # Calculate average train/val counts and ratios
    triplet_ratios = {}
    for name, counts in all_triplets.items():
        avg_train = np.mean(counts["train"])
        avg_val = np.mean(counts["val"])

        # Calculate ratio, skip if validation count is 0
        if avg_val > 0:
            ratio = avg_train / avg_val
            triplet_ratios[name] = {
                "ratio": ratio,
                "avg_train": avg_train,
                "avg_val": avg_val,
            }

    return triplet_ratios


def create_averaged_pie_charts(avg_instruments, avg_verbs, avg_targets):
    """Create three large pie charts showing average distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Instruments pie chart
    labels = list(avg_instruments.keys())
    sizes = list(avg_instruments.values())
    axes[0].pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        textprops={"fontsize": 12},
    )

    # Verbs pie chart
    labels = list(avg_verbs.keys())
    sizes = list(avg_verbs.values())
    axes[1].pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        textprops={"fontsize": 12},
    )

    # Targets pie chart
    labels = list(avg_targets.keys())
    sizes = list(avg_targets.values())
    axes[2].pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        textprops={"fontsize": 12},
    )

    plt.tight_layout()
    plt.savefig("surgical_data_averaged_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Parse the data
    data = parse_data_file("01_dataset_scripts/CholecT50_dataset_scripts/data.txt")

    # Calculate averages for instruments, verbs, targets
    avg_instruments, avg_verbs, avg_targets = calculate_averages(data)

    # Create averaged pie charts
    create_averaged_pie_charts(avg_instruments, avg_verbs, avg_targets)
    # Calculate triplet ratios
    triplet_ratios = calculate_triplet_ratios(data)

    # Create performance analysis
    # create_performance_analysis(triplet_ratios)


if __name__ == "__main__":
    main()

    # mAP data from the table
    map_scores = {
        "grasper,grasp,specimen_bag": 0.9352,
        "grasper,retract,gallbladder": 0.9037,
        "grasper,retract,liver": 0.8640,
        "irrigator,aspirate,fluid": 0.8650,
        "hook,dissect,gallbladder": 0.8629,
        "clipper,clip,cystic_duct": 0.8046,
        "clipper,clip,cystic_artery": 0.7618,
        "scissors,cut,cystic_duct": 0.6876,
        "scissors,cut,cystic_artery": 0.6871,
        "hook,retract,gallbladder": 0.6353,
        "hook,dissect,omentum": 0.6395,
        "bipolar,coagulate,abdominal_wall_cavity": 0.6206,
        "grasper,retract,omentum": 0.6035,
        "irrigator,irrigate,liver": 0.5777,
        "bipolar,coagulate,omentum": 0.5649,
        "hook,dissect,cystic_duct": 0.5161,
        "irrigator,dissect,cystic_pedicle": 0.5000,
        "hook,null_verb,null_target": 0.4846,
        "grasper,null_verb,null_target": 0.4672,
        "grasper,pack,gallbladder": 0.4426,
        "bipolar,coagulate,cystic_pedicle": 0.4106,
        "bipolar,coagulate,gallbladder": 0.4047,
        "hook,coagulate,gallbladder": 0.3914,
        "hook,coagulate,omentum": 0.3783,
        "scissors,null_verb,null_target": 0.3620,
        "bipolar,null_verb,null_target": 0.3611,
        "irrigator,null_verb,null_target": 0.3585,
        "grasper,retract,peritoneum": 0.3534,
        "grasper,grasp,gallbladder": 0.2692,
        "irrigator,retract,liver": 0.2778,
        "hook,coagulate,liver": 0.2669,
        "hook,dissect,cystic_artery": 0.2659,
        "hook,dissect,peritoneum": 0.2398,
        "bipolar,coagulate,cystic_artery": 0.2221,
        "clipper,null_verb,null_target": 0.2203,
        "bipolar,dissect,cystic_duct": 0.1944,
        "grasper,retract,gut": 0.1724,
        "bipolar,coagulate,cystic_plate": 0.1701,
        "scissors,dissect,gallbladder": 0.1667,
        "irrigator,irrigate,abdominal_wall_cavity": 0.1572,
        "irrigator,retract,gallbladder": 0.1500,
        "hook,retract,liver": 0.1494,
        "hook,coagulate,cystic_duct": 0.1329,
        "clipper,clip,blood_vessel": 0.1225,
        "grasper,grasp,cystic_duct": 0.1113,
        "scissors,cut,liver": 0.0905,
        "grasper,retract,cystic_plate": 0.0850,
        "bipolar,coagulate,blood_vessel": 0.0851,
        "scissors,dissect,omentum": 0.0840,
        "bipolar,dissect,cystic_artery": 0.0754,
        "irrigator,retract,omentum": 0.0653,
        "bipolar,retract,liver": 0.0567,
        "bipolar,retract,gallbladder": 0.0507,
        "scissors,cut,blood_vessel": 0.0445,
        "bipolar,dissect,omentum": 0.0403,
        "scissors,coagulate,omentum": 0.0345,
        "irrigator,dissect,omentum": 0.0206,
        "scissors,cut,peritoneum": 0.0183,
        "grasper,grasp,omentum": 0.0179,
        "grasper,retract,cystic_duct": 0.0153,
        "grasper,dissect,gallbladder": 0.0099,
        "grasper,grasp,cystic_plate": 0.0091,
        "bipolar,grasp,liver": 0.0062,
        "grasper,dissect,cystic_plate": 0.0013,
    }
