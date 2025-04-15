import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

class Utils:
    def __init__(self, primary_path = ""):
        self.seed = 42
        self.primary_path = primary_path
        self.subjects = sorted(glob(f"{self.primary_path}/kfall-dataset/sensor_data_new/SA*"))
        self.labels = sorted(glob(f"{self.primary_path}/kfall-dataset/label_data_new/*"))
        self.falls = [i for i in range(20, 35, 1)]
        self.dataset = {}
        self.sequence_threshold = 50
        self.task_dict = self.load_task_dict()

        self.read_dataset()
        
        os.makedirs('fall-guard', exist_ok=True)
        os.makedirs('fall-guard/dataset-insights', exist_ok=True)
        os.makedirs('fall-guard/dataset', exist_ok=True)
        os.makedirs('fall-guard/dataset/adl', exist_ok=True)
        os.makedirs('fall-guard/dataset/fall', exist_ok=True)
        os.makedirs('fall-guard/models', exist_ok=True)

    def load_task_dict(self):
        with open(f"{self.primary_path}/task_dict.json", "r") as f:
            return json.load(f)

    def read_dataset(self):
        print("\nStarting Read Dataset process...")
        for subject_no, subject in enumerate(self.subjects):
            self.dataset[subject[-4:]] = {}
            print(f"Processing Subject {subject[-4:]} ({subject_no + 1}/32).")
            readings = sorted(glob(f"{subject}/S*"))
            subject_labels = pd.read_excel(self.labels[subject_no])

            fall_data_index = 0
            for trial_reading in readings:
                temp = trial_reading.split("sensor_data_new")[1]

                if int(temp[10:12]) in self.falls:
                    label = [subject_labels.iat[fall_data_index, 3], subject_labels.iat[fall_data_index, 4]]
                    fall_data_index += 1
                else:
                    label = 0

                self.dataset[temp[1:5]][temp[6:15]] = {
                    "data": pd.read_csv(trial_reading),
                    "label": label
                }
        print("\nRead Dataset completed successfully.")

    def get_trial_lengths(self):
        data = defaultdict(list)

        for _, subject_readings in self.dataset.items():
            for task_name, task in subject_readings.items():
                data["Subject"].append(task_name)
                data["FrameCounts"].append(task["data"]['FrameCounter'].iat[-1])

        pd.DataFrame.from_dict(data).to_csv(f"{self.primary_path}/fall-guard/dataset-insights/subject_trials_frame_counts.csv")
        print("File to get trial lengths is saved successfully.")
        print(f"It is found in \"{self.primary_path}/fall-guard/dataset-insights/subject_trials_frame_counts.csv\"")

    def get_no_of_trials(self):
        data = {}

        for subject_no, subject_readings in self.dataset.items():
            task_info = defaultdict(int)
            for task_name in subject_readings:
                task_info[f"Task {int(task_name[4:6])}"] += 1
            data[subject_no] = task_info

        pd.DataFrame.from_dict(data, orient="index").fillna(0).to_csv(f"{self.primary_path}/fall-guard/dataset-insights/subject_trials.csv")
        print("File to get no of trials is saved successfully.")
        print(f"It is found in \"{self.primary_path}/fall-guard/dataset-insights/subject_trials.csv\"")

    def vector_summation(self):
        print("\nStarting vector summation process...")
        total_subjects = len(self.dataset)

        for subject_idx, (subject_no, subject_readings) in enumerate(self.dataset.items(), 1):
            print(f"Processing Subject {subject_no} ({subject_idx}/{total_subjects}).")

            for task in subject_readings:
                subject_readings[task]["data"]["Acc"] = np.sqrt(subject_readings[task]["data"]["AccX"]**2 + subject_readings[task]["data"]["AccY"]**2 + subject_readings[task]["data"]["AccZ"]**2)
                subject_readings[task]["data"]["Gyr"] = np.sqrt(subject_readings[task]["data"]["GyrX"]**2 + subject_readings[task]["data"]["GyrY"]**2 + subject_readings[task]["data"]["GyrZ"]**2)
                subject_readings[task]["data"]["Euler"] = np.sqrt(subject_readings[task]["data"]["EulerX"]**2 + subject_readings[task]["data"]["EulerY"]**2 + subject_readings[task]["data"]["EulerZ"]**2)
                self.dataset[subject_no][task]["data"] = subject_readings[task]["data"][["FrameCounter", "Acc", "Gyr", "Euler"]]
        print("\nVector summation completed successfully.")

    def plot_subject_readings(self):
        print("\nStarting Data plot process...")
        os.makedirs(f"{self.primary_path}/fall-guard/kfall-dataset-plots", exist_ok=True)
        total_subjects = len(self.dataset)

        for subject_idx, (subject_no, subject_readings) in enumerate(self.dataset.items(), 1):
            print(f"Processing Subject {subject_no} ({subject_idx}/{total_subjects}).")
            prev_task_id = ""
            for task, data in subject_readings.items():
                task_id = task[4:6]
                if task_id != prev_task_id:
                    prev_task_id = task_id
                    plt.figure(figsize=(12, 8))
                    plt.plot(data["data"]["FrameCounter"], data["data"]["Acc"], label="Gyroscope (°/s)")
                    plt.plot(data["data"]["FrameCounter"], data["data"]["Gyr"], label="Gyroscope (°/s)")
                    plt.plot(data["data"]["FrameCounter"], data["data"]["Euler"], label="Gyroscope (°/s)")

                    plt.xlabel("Frame Counter")
                    plt.ylabel("Gyroscope Reading")
                    plt.title(f"Gyroscope Readings Over Time for {self.task_dict[task_id]} by Subject {subject_no}")
                    plt.legend(loc="upper right")
                    plt.tight_layout()

                    plt.savefig(f"{self.primary_path}/fall-guard/kfall-dataset-plots/{task}.png")
                    plt.close()

        print("\nData plot completed successfully.")

    def normalize_data(self):
        print("\nStarting data normalization process...")
        scaler = MinMaxScaler()
        total_subjects = len(self.dataset)

        for subject_idx, (subject_no, subject_readings) in enumerate(self.dataset.items(), 1):
            print(f"Processing Subject {subject_no} ({subject_idx}/{total_subjects}).")

            for task_name, task_reading in subject_readings.items():
                frame_counter = task_reading['data']['FrameCounter'].values
                features = task_reading['data'][['Acc', 'Gyr', 'Euler']].values
                normalized_features = scaler.fit_transform(features)
                normalized_df = pd.DataFrame(
                    data=np.column_stack([frame_counter, normalized_features]),
                    columns=['FrameCounter', 'Acc', 'Gyr', 'Euler']
                )
                self.dataset[subject_no][task_name]["data"] = normalized_df

        print("\nData normalization completed successfully.")

    def segments_formation(self):
        print("\nStarting segments formation process...")
        total_subjects = len(self.dataset)

        for subject_idx, (subject_no, subject_readings) in enumerate(self.dataset.items(), 1):
            print(f"Processing Subject {subject_no} ({subject_idx}/{total_subjects}).")

            for task_name, task_reading in subject_readings.items():

                temp_data = task_reading["data"]
                temp_label = task_reading["label"]
                segments = {"data": [], "label": []}

                if temp_label == 0:
                    for i in range(0, len(temp_data), self.sequence_threshold):
                        segments["data"].append(temp_data.iloc[i:i+self.sequence_threshold])
                        segments["label"].append(0)

                    if len(segments["data"][-1]) < self.sequence_threshold // 2:
                        segments["data"].pop()
                    else:
                        r = segments["data"][-1].tail(5)
                        while len(segments["data"][-1]) < self.sequence_threshold:
                            segments["data"][-1] = pd.concat([segments["data"][-1], r]).iloc[:self.sequence_threshold]

                else:
                    for i in range(0, len(temp_data), self.sequence_threshold):
                        segments["data"].append(temp_data.iloc[i:i+self.sequence_threshold])
                        condition = (temp_label[0] <= segments["data"][-1].iat[0, 0] <= temp_label[1]) or (temp_label[0] <= segments["data"][-1].iat[-1, 0] <= temp_label[1])
                        segments["label"].append(1 if condition else 0)

                    if len(segments["data"][-1]) < 25:
                        segments["data"].pop()
                    else:
                        r = segments["data"][-1].tail(5)
                        while len(segments["data"][-1]) < self.sequence_threshold:
                            segments["data"][-1] = pd.concat([segments["data"][-1], r]).iloc[:self.sequence_threshold]

                self.dataset[subject_no][task_name] = segments
        print("\nSegments formation completed successfully.")