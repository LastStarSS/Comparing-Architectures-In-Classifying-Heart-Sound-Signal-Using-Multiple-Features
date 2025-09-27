import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models
from matplotlib.widgets import CheckButtons

# 1. Settings
DATASET_PATH = "D:/Coding/NCKH/Classification-Of-Heart-Sound-Signal-Using-Multiple-Features/Data"
LABELS = ["MR", "MS", "N", "MVP"]
N_MFCC = 16
N_FRAMES = 30
IMG_CHANNELS = 2
EPOCHS = 30
BATCH_SIZE = 32

# 2. Load dataset and extract MFCC + DWT features
features = []
labels = []

for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            y, sr = librosa.load(filepath, sr=None)

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            if mfcc.shape[1] < N_FRAMES:
                continue
            mfcc_feat = mfcc[:, :N_FRAMES]

            # DWT
            coeffs = pywt.wavedec(y, 'db4', level=4)
            dwt_feat = coeffs[0]  # approximation
            if len(dwt_feat) < N_FRAMES:
                dwt_feat = np.pad(dwt_feat, (0, N_FRAMES - len(dwt_feat)))
            else:
                dwt_feat = dwt_feat[:N_FRAMES]
            dwt_feat = dwt_feat.reshape(1, N_FRAMES)

            # Repeat DWT to match MFCC shape
            dwt_feat_rep = np.repeat(dwt_feat, N_MFCC, axis=0)

            # Stack MFCC + DWT as channels
            combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)  # (n_mfcc, n_frames, 2)
            features.append(combined_feat)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

# One-hot encode labels
encoder = LabelBinarizer()
y_onehot = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42, stratify=y
)

# 3. Model builders
def build_cnn(input_shape=(N_MFCC, N_FRAMES, IMG_CHANNELS), n_classes=len(LABELS)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape=(N_FRAMES, N_MFCC * IMG_CHANNELS), n_classes=len(LABELS)):
    model = models.Sequential([
        layers.LSTM(128, input_shape=input_shape, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rcnn(input_shape=(N_MFCC, N_FRAMES, IMG_CHANNELS), n_classes=len(LABELS)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Reshape((-1, 64)),  # flatten spatial features into sequence
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare reshaped inputs
X_train_lstm = X_train.reshape((X_train.shape[0], N_FRAMES, N_MFCC * IMG_CHANNELS))
X_test_lstm  = X_test.reshape((X_test.shape[0], N_FRAMES, N_MFCC * IMG_CHANNELS))

# 4. Train models
histories = {}

print("Training CNN...")
cnn_model = build_cnn()
histories["CNN"] = cnn_model.fit(
    X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test), verbose=1
)

print("Training LSTM...")
lstm_model = build_lstm()
histories["LSTM"] = lstm_model.fit(
    X_train_lstm, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_test_lstm, y_test), verbose=1
)

print("Training RCNN...")
rcnn_model = build_rcnn()
histories["RCNN"] = rcnn_model.fit(
    X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test), verbose=1
)

# 5. Plot comparisons
def plot_histories(histories):
    plt.figure(figsize=(14,6))

    # Assign a color per model
    colors = {
        "CNN": "tab:blue",
        "LSTM": "tab:green",
        "RCNN": "tab:red"
    }

    # Store plotted line objects
    lines = {}

    # Accuracy subplot
    ax1 = plt.subplot(1,2,1)
    for name, hist in histories.items():
        l1, = ax1.plot(hist.history['accuracy'], color=colors[name], linestyle='-', label=f'{name} Train')
        l2, = ax1.plot(hist.history['val_accuracy'], color=colors[name], linestyle='--', label=f'{name} Val')
        lines[f"{name} Train"] = l1
        lines[f"{name} Val"] = l2
    ax1.set_title("Accuracy Comparison")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0,1)
    ax1.legend()

    # Loss subplot
    ax2 = plt.subplot(1,2,2)
    for name, hist in histories.items():
        l3, = ax2.plot(hist.history['loss'], color=colors[name], linestyle='-', label=f'{name} Train')
        l4, = ax2.plot(hist.history['val_loss'], color=colors[name], linestyle='--', label=f'{name} Val')
        lines[f"{name} Train (Loss)"] = l3
        lines[f"{name} Val (Loss)"] = l4
    ax2.set_title("Loss Comparison")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()

    # Create checkboxes for toggling
    rax = plt.axes([0.01, 0.4, 0.15, 0.4])  # position of checkbox panel
    labels = list(lines.keys())
    visibility = [line.get_visible() for line in lines.values()]
    check = CheckButtons(rax, labels, visibility)

    def toggle(label):
        line = lines[label]
        line.set_visible(not line.get_visible())
        plt.draw()

    check.on_clicked(toggle)
    plt.show()


plot_histories(histories)
