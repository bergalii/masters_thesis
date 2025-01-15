class MultiLabelFrameDataset(Dataset):
    """
    Dataset class for handling frame-based data with multi-label annotations
    """

    def __init__(
        self,
        frames_dir: str,
        annotations_dir: str,
        max_clip_length: int,
        train: bool = True,
    ):

        self.frames_dir = frames_dir
        self.annotations_dir = annotations_dir
        self.max_clip_length = max_clip_length
        self.train = train

        # Initialize transforms
        self.preprocess = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.3656, 0.3660, 0.3670], std=[0.2025, 0.2021, 0.2027]),
            ]
        )

        # Augmentation pipeline
        self.transform = A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.05, 0.05),
                            contrast_limit=(-0.05, 0.05),
                            p=0.5,
                        ),
                    ],
                    p=0.3,
                ),
                A.CLAHE(clip_limit=(1, 1.1), tile_grid_size=(6, 6), p=0.3),
                A.AdvancedBlur(blur_limit=(3, 7), p=0.3),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.6,
                ),
            ]
        )

        # Process all annotations and frames
        self.clips, self.labels = self._generate_clips()
        self.transition_matrix = self._generate_transition_matrix()
        self.action_durations = self._calculate_action_durations()

    def _generate_clips(self) -> Tuple[List[List[str]], List[torch.Tensor]]:
        """
        Process the dataset to create clips based on action changes.
        Returns:
            clips: List of frame paths for each clip
            labels: List of multi-hot encoded labels for each clip
        """
        # Get all XML annotation files
        all_annotations = sorted(
            [f for f in os.listdir(self.annotations_dir) if f.endswith(".xml")]
        )
        print(f"{len(all_annotations)} frames in total.")
        clips = []
        labels = []
        current_clip = []
        current_labels = None

        for annotation_file in all_annotations:
            xml_path = os.path.join(self.annotations_dir, annotation_file)

            # Parse XML to get filename and multi-hot encoded labels
            frame_filename, frame_labels = self._parse_xml(xml_path)
            frame_path = os.path.join(self.frames_dir, frame_filename)

            if current_labels is None:
                current_labels = frame_labels
                current_clip.append(frame_path)
            elif not torch.equal(current_labels, frame_labels):
                # Action change detected, save current clip
                clips.append(current_clip)
                labels.append(current_labels)
                # Start new clip
                current_clip = [frame_path]
                current_labels = frame_labels
            else:
                current_clip.append(frame_path)

            # For debugging
            if len(clips) == 10:
                print(f"Generated {len(clips)} clips.")
                return clips, labels

        if len(current_clip) > 0:
            clips.append(current_clip)
            labels.append(current_labels)
        print(f"Generated {len(clips)} clips.")
        return clips, labels

    def _parse_xml(self, xml_path: str) -> Tuple[str, torch.Tensor]:
        """
        Parse XML annotation file and return the frame filename and multi-hot encoded labels.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract filename
        filename = root.find("filename").text

        # Initialize multi-hot vector
        labels = torch.zeros(NUM_CLASSES)

        # Set 1 for each action present in the frame
        for obj in root.findall("object"):
            action = obj.find("name").text
            if action in CLASSES:
                labels[CLASSES.index(action)] = 1

        return filename, labels

    def _apply_augmentations(self, frames: np.ndarray) -> np.ndarray:
        """Apply the same augmentation transform to all frames in the clip."""

        # Get initial transform params from first frame
        data = self.transform(image=frames[0])

        # Apply same transform to all frames
        augmented_frames = []
        for frame in frames:
            augmented = A.ReplayCompose.replay(data["replay"], image=frame)
            augmented_frames.append(augmented["image"])

        return np.stack(augmented_frames)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_paths = self.clips[idx]
        label = self.labels[idx]
        # If clip length exceeds max_clip_length, sample uniformly
        if len(clip_paths) > self.max_clip_length:
            indices = np.linspace(0, len(clip_paths) - 1, self.max_clip_length).astype(
                int
            )
            clip_paths = [clip_paths[i] for i in indices]

        # Load all frames
        frames = []
        for path in clip_paths:
            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (279, 224))
            frames.append(frame)

        frames = np.stack(frames)  # Shape: (T, H, W, C)

        # Apply augmentations across frames (only to train set)
        if self.train:
            frames = self._apply_augmentations(frames)
        # Load and preprocess frames
        frames = torch.stack([self.preprocess(frame) for frame in frames])
        # Transpose to (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        return frames, label, len(clip_paths)

    def _generate_transition_matrix(self):
        """
        Generate transition matrix between clips. Each entry (i,j) represents
        the probability of transitioning from action i to action j between consecutive clips.
        """
        # Initialize transition matrix for all possible action pairs
        transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

        # For each pair of consecutive clips in our dataset
        for clip_idx in range(len(self.labels) - 1):
            # Get labels for current and next clip
            current_clip_labels = self.labels[clip_idx]
            next_clip_labels = self.labels[clip_idx + 1]

            # Get indices of active actions (where label is 1) in both clips
            current_actions = current_clip_labels.nonzero().view(-1)
            next_actions = next_clip_labels.nonzero().view(-1)

            # For each pair of actions between current and next clip
            for current_action in current_actions:
                for next_action in next_actions:
                    # Increment the transition count from current_action to next_action
                    transition_matrix[current_action.item(), next_action.item()] += 1

        # Normalize each row to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)

        # If a row sums to 0 (action never appears), that row will be all zeros
        transition_matrix = np.divide(
            transition_matrix,
            row_sums,
            out=np.zeros_like(transition_matrix),
            where=row_sums != 0,
        )

        return transition_matrix

    def _calculate_action_durations(self) -> Dict[str, float]:
        """
        Calculate average duration for each unique label combination
        Returns:
            Dict mapping label combination tuple to average duration in frames
        """
        # Dictionary to store durations for each label combination
        duration_stats = defaultdict(list)

        current_labels = None
        current_start = 0
        current_length = 0

        # Iterate through all clips and their lengths
        for clip_idx, (clip_paths, label) in enumerate(zip(self.clips, self.labels)):
            label_tuple = tuple(
                label.numpy().astype(int)
            )  # Convert to tuple for hashable key
            duration_stats[label_tuple].append(len(clip_paths))

        # Calculate average duration for each label combination
        average_durations = {}
        for label_combo, durations in duration_stats.items():
            average_durations[label_combo] = sum(durations) / len(durations)
        return average_durations
