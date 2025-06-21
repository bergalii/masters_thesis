import pandas as pd
import ast

VERB_MAPPINGS = {
    "Hold": "Manipulation",
    "Pull": "Manipulation",
    "Push": "Manipulation",
    "Travel": "Inactive",
    "Still": "Inactive",
}

TARGET_MAPPINGS = {
    "Bladder": "Bladder_Region",
    "Bladder Neck": "Bladder_Region",
    "Prostate": "Prostate_Region",
    "Prostate Apex": "Prostate_Region",
}

INSTRUMENT_MAPPINGS = {
    "Bipolar Forceps": "Bipolar_Tools",
    "Bipolar Maryland Forceps": "Bipolar_Tools",
}


def modify_csv(input_path, output_path):
    # Read the CSV
    df = pd.read_csv(input_path)

    # Create global mappings to track unique components and their IDs
    global_instrument_mapping = {}
    global_verb_mapping = {}
    global_target_mapping = {}
    global_action_mapping = {}

    # First pass: collect all unique components after mapping
    unique_instruments = set()
    unique_verbs = set()
    unique_targets = set()
    unique_actions = set()

    for _, row in df.iterrows():
        instrument_names = ast.literal_eval(row["instrument_label_names"])
        verb_names = ast.literal_eval(row["verb_label_names"])
        target_names = ast.literal_eval(row["target_label_names"])
        action_names = ast.literal_eval(row["action_label_names"])

        # Apply mappings and collect unique values
        for inst in instrument_names:
            mapped_inst = INSTRUMENT_MAPPINGS.get(inst, inst)
            unique_instruments.add(mapped_inst)

        for verb in verb_names:
            mapped_verb = VERB_MAPPINGS.get(verb, verb)
            unique_verbs.add(mapped_verb)

        for target in target_names:
            mapped_target = TARGET_MAPPINGS.get(target, target)
            unique_targets.add(mapped_target)

        for action in action_names:
            parts = action.split(",")
            mapped_inst = INSTRUMENT_MAPPINGS.get(parts[0], parts[0])
            mapped_verb = VERB_MAPPINGS.get(parts[1], parts[1])
            mapped_target = TARGET_MAPPINGS.get(parts[2], parts[2])
            mapped_action = f"{mapped_inst},{mapped_verb},{mapped_target}"
            unique_actions.add(mapped_action)

    # Create ID mappings for unique components
    for i, inst in enumerate(sorted(unique_instruments)):
        global_instrument_mapping[inst] = i

    for i, verb in enumerate(sorted(unique_verbs)):
        global_verb_mapping[verb] = i

    for i, target in enumerate(sorted(unique_targets)):
        global_target_mapping[target] = i

    for i, action in enumerate(sorted(unique_actions)):
        global_action_mapping[action] = i

    # Second pass: update all rows with new mappings
    new_rows = []

    for _, row in df.iterrows():
        new_row = row.copy()

        # Parse current values
        instrument_names = ast.literal_eval(row["instrument_label_names"])
        verb_names = ast.literal_eval(row["verb_label_names"])
        target_names = ast.literal_eval(row["target_label_names"])
        action_names = ast.literal_eval(row["action_label_names"])

        # Apply mappings and get new names (removing duplicates)
        new_instrument_names = []
        new_instrument_labels = []
        seen_instruments = set()

        for inst in instrument_names:
            mapped_inst = INSTRUMENT_MAPPINGS.get(inst, inst)
            if mapped_inst not in seen_instruments:
                new_instrument_names.append(mapped_inst)
                new_instrument_labels.append(global_instrument_mapping[mapped_inst])
                seen_instruments.add(mapped_inst)

        new_verb_names = []
        new_verb_labels = []
        seen_verbs = set()

        for verb in verb_names:
            mapped_verb = VERB_MAPPINGS.get(verb, verb)
            if mapped_verb not in seen_verbs:
                new_verb_names.append(mapped_verb)
                new_verb_labels.append(global_verb_mapping[mapped_verb])
                seen_verbs.add(mapped_verb)

        new_target_names = []
        new_target_labels = []
        seen_targets = set()

        for target in target_names:
            mapped_target = TARGET_MAPPINGS.get(target, target)
            if mapped_target not in seen_targets:
                new_target_names.append(mapped_target)
                new_target_labels.append(global_target_mapping[mapped_target])
                seen_targets.add(mapped_target)

        # Handle actions (triplets)
        new_action_names = []
        new_action_labels = []

        for action in action_names:
            parts = action.split(",")
            mapped_inst = INSTRUMENT_MAPPINGS.get(parts[0], parts[0])
            mapped_verb = VERB_MAPPINGS.get(parts[1], parts[1])
            mapped_target = TARGET_MAPPINGS.get(parts[2], parts[2])
            mapped_action = f"{mapped_inst},{mapped_verb},{mapped_target}"

            new_action_names.append(mapped_action)
            new_action_labels.append(global_action_mapping[mapped_action])

        # Update row with new values
        new_row["instrument_label_names"] = str(new_instrument_names)
        new_row["instrument_label"] = str(new_instrument_labels)
        new_row["verb_label_names"] = str(new_verb_names)
        new_row["verb_label"] = str(new_verb_labels)
        new_row["target_label_names"] = str(new_target_names)
        new_row["target_label"] = str(new_target_labels)
        new_row["action_label_names"] = str(new_action_names)
        new_row["action_label"] = str(new_action_labels)

        new_rows.append(new_row)

    # Create new dataframe
    new_df = pd.DataFrame(new_rows)

    # Save to output path
    new_df.to_csv(output_path, index=False)

    # Print summary of changes
    print("Component mappings applied:")
    print(
        f"Instruments: {len(unique_instruments)} unique (was {len(set().union(*[set(ast.literal_eval(row['instrument_label_names'])) for _, row in df.iterrows()]))})"
    )
    print(
        f"Verbs: {len(unique_verbs)} unique (was {len(set().union(*[set(ast.literal_eval(row['verb_label_names'])) for _, row in df.iterrows()]))})"
    )
    print(
        f"Targets: {len(unique_targets)} unique (was {len(set().union(*[set(ast.literal_eval(row['target_label_names'])) for _, row in df.iterrows()]))})"
    )
    print(
        f"Actions: {len(unique_actions)} unique (was {len(set().union(*[set(ast.literal_eval(row['action_label_names'])) for _, row in df.iterrows()]))})"
    )

    return new_df


# Usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    input_csv = (
        "/data/Berk/masters_thesis/05_datasets_dir/UKE/gt.csv"  # Your original CSV file
    )
    output_csv = "annotations_combined.csv"  # New CSV file with combined components

    modified_df = modify_csv(input_csv, output_csv)
    print(f"\nModified CSV saved to: {output_csv}")
