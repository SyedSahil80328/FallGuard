import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
from imblearn.over_sampling import SMOTE
import random

class FallDetectionUtilities:
    def __init__ (self, raw_data):
        self.raw_data = raw_data
        self.shaped_data = [[], []]
        self.segments = ""
        self.adl_count = 0
        self.fall_count = 0
        self.frame_counter_starts = []

    def display_label_counts(self):
        self.segments = Counter(self.shaped_data[1])
        self.adl_count = self.segments[0]
        self.fall_count = self.segments[1]
        self.segments = self.adl_count + self.fall_count
        return f"Total Segments: {self.segments}, ADL Segments (Class label 0): {self.adl_count}, Fall Segments (Class label 1): {self.fall_count}."

    def reshape_dataset(self):
        print("\nStarting dataset reshaping...")
        for subject_no, subject_readings in self.raw_data.items():
            print(f"Processing Subject {subject_no}.")
            for task_name, task_readings in subject_readings.items():
                segments_count = len(task_readings["data"])
                for task_index in range(segments_count):
                    self.shaped_data[0].append(task_readings["data"][task_index])
                    self.shaped_data[1].append(task_readings["label"][task_index])
                    if task_readings["label"][task_index] == 1:
                        self.frame_counter_starts.append(task_readings["data"][task_index]['FrameCounter'].iloc[0])
        print(f"\nDataset reshaping completed!")
        print(self.display_label_counts())

    def apply_smote(self, time_steps=50, selected_columns=['Acc', 'Gyr', 'Euler'],seed=42):
        X, y = [], self.shaped_data[1]
        for segment in self.shaped_data[0]:
            X.append(segment[selected_columns].values.flatten())
        X, y = np.array(X), np.array(y)

        smote = SMOTE(random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        num_original = len(y)
        num_generated = len(X_resampled) - num_original

        synthetic_segments = []
        for i in range(num_generated):
            synthetic_flat = X_resampled[num_original + i]
            synthetic_matrix = synthetic_flat.reshape(time_steps, len(selected_columns))
            start_fc = random.choice(self.frame_counter_starts)
            frame_counter = np.arange(start_fc, start_fc + time_steps)
            df_synthetic = pd.DataFrame(synthetic_matrix, columns=selected_columns)
            df_synthetic.insert(0, 'FrameCounter', frame_counter)
            synthetic_segments.append(df_synthetic)

        return synthetic_segments

    def train_lstm_discriminator_on_falls(self, real_samples, fake_samples, selected_columns=['Acc', 'Gyr', 'Euler'], hidden_size=64, epochs=15, batch_size=32, seed=42):
        dataset = FallDataset(real_samples, fake_samples)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = LSTMDiscriminator(hidden_size=hidden_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(epochs):
            model.train()
            total_loss, preds_epoch, labels_epoch = 0, [], []
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds_epoch += (outputs > 0.5).cpu().numpy().tolist()
                labels_epoch += y.cpu().numpy().tolist()

            acc = accuracy_score(labels_epoch, preds_epoch)
            prec = precision_score(labels_epoch, preds_epoch, zero_division=0)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                all_preds += (preds > 0.5).cpu().numpy().tolist()
                all_labels += y.cpu().numpy().tolist()

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        fp_score = fp / (fp + tn + 1e-6)
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Acc: {acc * 100:.2f}%, Prec: {prec * 100:.2f}%")
        return model, fp_score

    def plot_confusion_matrix(self, true_labels, pred_labels, class_names=["Fake", "Real"]):
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = hn[-1]
        return self.fc(out).squeeze()

class FallDataset(torch.utils.data.Dataset):
    def __init__(self, real_segments, fake_segments, selected_columns=['Acc', 'Gyr', 'Euler']):
        self.data = [df[selected_columns].values.astype(np.float32) for df in (real_segments + fake_segments)]
        self.labels = [1] * len(real_segments) + [0] * len(fake_segments)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)