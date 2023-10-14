class LabelMapper:
    def __init__(self, index_to_label=[]):
        self.index_to_label = index_to_label
        self.label_to_index = {}
        for i in range(len(self.index_to_label)):
            label = self.index_to_label[i]
            self.label_to_index[label] = i

    def num_labels(self):
        return len(self.index_to_label)

    def maybe_add_label(self, label):
        if label not in self.label_to_index:
            label_idx = len(self.label_to_index)
            self.label_to_index[label] = label_idx
            self.index_to_label.append(label)

    def index_for_label(self, label):
        return self.label_to_index[label]

    def label_for_index(self, i):
        return self.index_to_label[i]


