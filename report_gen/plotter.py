import json
import os

import matplotlib.pyplot as plt

class JSONDataPlotter:
    FILE_1 = "sota"
    FILE_2 = "novel"
    ORIGINAL_ACC = "original_acc"
    ADAPTED_ACC = "adapted_acc"
    PARTIAL_ACC = "partial_acc"

    def __init__(self, reports_dir, dataset, batch_size, n_drop_classes, severity):
        self.reports_dir = reports_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_drop_classes = n_drop_classes
        self.severity = severity

        self._data1 = None
        self._data2 = None

        self._get_avg_data()

    def _get_data_dir(self):
        data_dir = f"{self.reports_dir}/{self.dataset}/bs{self.batch_size}_dc{self.n_drop_classes}_s{self.severity}"

        if not os.path.isdir(data_dir):
            raise Exception(f"Directory {data_dir} does not exist")
        
        return data_dir
    
    def _get_avg_data(self):
        # average all the data accross seeds
        seed_dirs = os.listdir(self._get_data_dir())
        if len(seed_dirs) == 0:
            raise Exception(f"No seed directories found in {self._get_data_dir()}")
        
        data_list1 = []
        data_list2 = []

        for seed_dir in seed_dirs:
            file_1 = f"{self._get_data_dir()}/{seed_dir}/{self.FILE_1}.json"
            file_2 = f"{self._get_data_dir()}/{seed_dir}/{self.FILE_2}.json"

            with open(file_1) as f1:
                data1 = json.load(f1)
                data_list1.append(data1)
            with open(file_2) as f2:
                data2 = json.load(f2)
                data_list2.append(data2)
        
        self._data1 = self._avg_values(data_list1)
        self._data2 = self._avg_values(data_list2)

    def _avg_values(self, data_list):
        """
        Get the avg value of all the data fields across multiple files
        """
        avg_data = {}

        for data in data_list:
            for key, value in data.items():
                if key not in avg_data:
                    avg_data[key] = {}

                for sub_key, sub_value in value.items():
                    if sub_key not in avg_data[key]:
                        avg_data[key][sub_key] = 0

                    avg_data[key][sub_key] += sub_value

        n_files = len(data_list)
        
        for key, value in avg_data.items():
            for sub_key, sub_value in value.items():
                avg_data[key][sub_key] = sub_value / n_files

        return avg_data
                    
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