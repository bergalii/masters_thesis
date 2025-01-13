# def _generate_transition_matrix(self):
#   # Initialize the transition matrix
#   transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
#   # Get all XML annotation files
#   all_annotations = sorted([f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')])
#   # Initialize variables to store previous frame's labels
#   prev_frame_labels = None

#   i = 0
#   # Iterate through the annotation files
#   for annotation_file in all_annotations:
#       if i == 10:
#         break
#       i += 1
#       xml_path = os.path.join(self.annotations_dir, annotation_file)
#       # Parse XML to get filename and multi-hot encoded labels
#       frame_filename, frame_labels = self._parse_xml(xml_path)
#       if prev_frame_labels is not None:
#         # Get indices where labels are 1
#         prev_indices = prev_frame_labels.nonzero().view(-1)
#         curr_indices = frame_labels.nonzero().view(-1)

#         # Update the transition matrix
#         for prev_idx in prev_indices:
#           for curr_idx in curr_indices:
#               transition_matrix[prev_idx.item(), curr_idx.item()] +=1

#       prev_frame_labels = frame_labels

#   # Normalize each row of the transition matrix
#   row_sums = transition_matrix.sum(axis=1, keepdims=True)
#   transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)
#   return transition_matrix
