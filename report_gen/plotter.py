import json
import matplotlib.pyplot as plt

class JSONDataPlotter:
    def __init__(self, file1, file2):
        """
        Parameters
        ----------
        file1 : str : Path to outputs of SOTA environment
        file2 : str : Path to outputs of our environment
        """
        self.file1 = file1
        self.file2 = file2

        self._data1, self._data2 = self._load_data()

    def _load_data(self):
        # Load data from JSON files
        with open(self.file1) as f1:
            data1 = json.load(f1)
        with open(self.file2) as f2:
            data2 = json.load(f2)
        return data1, data2

    def plot_data_for_all_labels(self):

        # Extract x and y values from data
        x1 = list(self._data1["adapted_acc"].keys())
        y1 = list(self._data1["adapted_acc"].values())
        x2 = list(self._data2["adapted_acc"].keys())
        y2 = list(self._data2["adapted_acc"].values())
        x3 = list(self._data1["original_acc"].keys())
        y3 = list(self._data1["original_acc"].values())

        # Calculate the width of each bar
        bar_width = 0.25

        # Adjust the x-coordinates of the bars
        x1_adjusted = [x - bar_width for x in range(len(x1))]
        x2_adjusted = [x for x in range(len(x2))]
        x3_adjusted = [x + bar_width for x in range(len(x3))]

        # Plot the data
        plt.bar(x1_adjusted, y1, width=bar_width, label='balanced')
        plt.bar(x2_adjusted, y2, width=bar_width, label='imbalanced')
        plt.bar(x3_adjusted, y3, width=bar_width, label='without adaptation')

        # Add labels and title
        plt.xlabel('Corruptions')
        plt.ylabel('Accuracy')

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

        # Set the x-ticks to be the corruptions with vertical orientation
        plt.xticks(x2_adjusted, x2, rotation='vertical')

        # Show the plot
        plt.show()

    def plot_data_for_unseen_labels(self):
        x1 = list(self._data1["partial_acc"].keys())
        y1 = list(self._data1["partial_acc"].values())
        x2 = list(self._data2["partial_acc"].keys())
        y2 = list(self._data2["partial_acc"].values())

        bar_width = 0.25
        x1_adjusted = [x - bar_width for x in range(len(x1))]
        x2_adjusted = [x for x in range(len(x2))]

        plt.bar(x1_adjusted, y1, width=bar_width, label='balanced')
        plt.bar(x2_adjusted, y2, width=bar_width, label='imbalanced')

        plt.xlabel('Corruptions')
        plt.ylabel('Accuracy')

        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

        plt.xticks(x2_adjusted, x2, rotation='vertical')

        plt.show()