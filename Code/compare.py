import os
import numpy as np
import librosa
import pywt
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class HeartSoundCompare:
    def __init__(self, args):
        self.data_path = args.data_path
        self.labels = args.labels
        self.n_mfcc = args.n_mfcc
        self.n_frames = args.n_frames
        self.img_channels = args.img_channels
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate
        self.test_size = args.test_size
        self.model_save_dir = args.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_data(self):
        features, file_labels = [], []
        for label in self.labels:
            folder = os.path.join(self.data_path, label)
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    try:
                        filepath = os.path.join(folder, file)
                        y, sr = librosa.load(filepath, sr=None)

                        # MFCC
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
                        if mfcc.shape[1] < self.n_frames:
                            continue
                        mfcc_feat = mfcc[:, :self.n_frames]

                        # DWT
                        coeffs = pywt.wavedec(y, 'db4', level=4)
                        dwt_feat = coeffs[0]
                        if len(dwt_feat) < self.n_frames:
                            dwt_feat = np.pad(dwt_feat, (0, self.n_frames - len(dwt_feat)))
                        else:
                            dwt_feat = dwt_feat[:self.n_frames]
                        dwt_feat = dwt_feat.reshape(1, self.n_frames)
                        dwt_feat_rep = np.repeat(dwt_feat, self.n_mfcc, axis=0)

                        # Stack as channels
                        combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)
                        features.append(combined_feat)
                        file_labels.append(label)
                    except:
                        continue
        X = np.array(features)
        y = np.array(file_labels)
        encoder = LabelBinarizer()
        y_onehot = encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=self.test_size, random_state=42, stratify=y
        )
        # LSTM reshaped data
        X_train_lstm = X_train.reshape((X_train.shape[0], self.n_frames, self.n_mfcc*self.img_channels))
        X_test_lstm  = X_test.reshape((X_test.shape[0], self.n_frames, self.n_mfcc*self.img_channels))
        return X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm, encoder

    def build_cnn(self, input_shape):
        model = Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            Dropout(self.dropout_rate),
            layers.Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            Dropout(self.dropout_rate),
            
            Flatten(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate/2),
            Dense(len(self.labels), activation='softmax')
        ])
        model.compile(optimizer=Adam(self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_lstm(self, input_shape):
        model = models.Sequential([
            layers.LSTM(128, input_shape=input_shape, return_sequences=True, name='LSTM_1'),
            layers.Dropout(0.2, name ='Dropout_1'),
            layers.LSTM(64, input_shape=input_shape, name='LSTM_2'),
            layers.Dropout(0.2, name ='Dropout_2'),
            Dense(64, activation='relu', name = 'Dense_1'),
            Dense(len(self.labels), activation='softmax', name = 'Output')
        ])
        model.compile(optimizer=Adam(self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_rcnn(self, input_shape):
        model = models.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            Dropout(self.dropout_rate),

            layers.Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            Dropout(self.dropout_rate),

            layers.Reshape((-1, 64)), 
            layers.SimpleRNN(64),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate/2),
            Dense(len(self.labels), activation='softmax')
        ])
        model.compile(optimizer=Adam(self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_and_save(self, model, X_train, y_train, X_test, y_test, name):
        model_path = os.path.join(self.model_save_dir, f"{name}.h5")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, 
                            validation_data=(X_test, y_test),
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            callbacks=[early_stop],
                            verbose=1)
        model.save(model_path)
        return history, model

    def plot_histories(self, histories):
        fig = plt.figure(figsize=(14,6))

        colors = {"CNN":"tab:blue","LSTM":"tab:green","RCNN":"tab:red"}
        lines = {}

        # Accuracy subplot
        ax1 = plt.subplot(1,2,1)
        for name,hist in histories.items():
            l1, = ax1.plot(hist.history['accuracy'], color=colors[name], linestyle='-', label=f'{name} Train')
            l2, = ax1.plot(hist.history['val_accuracy'], color=colors[name], linestyle='--', label=f'{name} Val')
            lines[f'{name} Train'] = l1
            lines[f'{name} Val'] = l2
        ax1.set_title("Accuracy Comparison")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0,1)
        ax1.legend()

        # Loss subplot
        ax2 = plt.subplot(1,2,2)
        for name,hist in histories.items():
            l3, = ax2.plot(hist.history['loss'], color=colors[name], linestyle='-', label=f'{name} Train')
            l4, = ax2.plot(hist.history['val_loss'], color=colors[name], linestyle='--', label=f'{name} Val')
            lines[f'{name} Train (Loss)'] = l3
            lines[f'{name} Val (Loss)'] = l4
        ax2.set_title("Loss Comparison")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0,1)
        ax2.legend()
        plt.savefig(os.path.join(self.model_save_dir,'comparison.png'))

        plt.tight_layout()

        # Checkbox widget
        rax = plt.axes([0.3, 0.1, 0.15, 0.4])  # position [left, bottom, width, height]
        check = CheckButtons(rax, list(lines.keys()), [line.get_visible() for line in lines.values()])

        def toggle(label):
            lines[label].set_visible(not lines[label].get_visible())
            plt.draw()

        check.on_clicked(toggle)
        plt.show()

    def visualize_samples(self, model, X_test_lstm, y_test, encoder, n=10):
        # Load the model if not already
        if isinstance(model, str):
            model = models.load_model(model)

        # Collect all test files with labels
        test_files = []
        for label in self.labels:
            folder = os.path.join(self.data_path, label)
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    test_files.append((os.path.join(folder, file), label))

        # Randomly select n files
        random_files = random.sample(test_files, n)

        plt.figure(figsize=(15, 10))
        for idx, (filepath, true_label) in enumerate(random_files, 1):
            y_audio, sr = librosa.load(filepath, sr=None)

            # MFCC + DWT features
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=self.n_mfcc)
            if mfcc.shape[1] < self.n_frames:
                continue
            mfcc_feat = mfcc[:, :self.n_frames]

            coeffs = pywt.wavedec(y_audio, 'db4', level=4)
            dwt_feat = coeffs[0]
            if len(dwt_feat) < self.n_frames:
                dwt_feat = np.pad(dwt_feat, (0, self.n_frames - len(dwt_feat)))
            else:
                dwt_feat = dwt_feat[:self.n_frames]
            dwt_feat = dwt_feat.reshape(1, self.n_frames)
            dwt_feat_rep = np.repeat(dwt_feat, self.n_mfcc, axis=0)

            combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)  # (16,30,2)
            combined_feat = combined_feat[np.newaxis, ...]
            lstm_input = combined_feat.reshape(1, self.n_frames, self.n_mfcc*self.img_channels)  # (1,30,32)
            pred_prob = model.predict(combined_feat)
            pred_label = encoder.classes_[np.argmax(pred_prob)]

            pred_label = encoder.classes_[np.argmax(pred_prob)]

            # CWT visualization
            coeffs_cwt, freqs = pywt.cwt(y_audio, scales=np.arange(1,128), wavelet='morl')
            plt.subplot(2, 5, idx)
            plt.imshow(np.abs(coeffs_cwt), aspect='auto', cmap='magma', origin='lower')
            plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
            plt.xlabel("Time")
            plt.ylabel("Scale")
            plt.tight_layout()

        plt.show()

    def main(self):
        X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm, encoder = self.load_data()
        histories = {}

        # CNN
        cnn_model = self.build_cnn((self.n_mfcc, self.n_frames, self.img_channels))
        histories['CNN'], _ = self.train_and_save(cnn_model, X_train, y_train, X_test, y_test, 'CNN')

        # LSTM
        lstm_model = self.build_lstm((self.n_frames, self.n_mfcc*self.img_channels))
        histories['LSTM'], _ = self.train_and_save(lstm_model, X_train_lstm, y_train, X_test_lstm, y_test, 'LSTM')

        # RCNN
        rcnn_model = self.build_rcnn((self.n_mfcc, self.n_frames, self.img_channels))
        histories['RCNN'], _ = self.train_and_save(rcnn_model, X_train, y_train, X_test, y_test, 'RCNN')

        self.plot_histories(histories)
        return histories

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Heart Sound Models")
    parser.add_argument('--data_path', type=str, default="D:/Coding/NCKH/Classification-Of-Heart-Sound-Signal-Using-Multiple-Features/Data")
    parser.add_argument('--labels', type=list, default=["MR","MS","N","MVP","AS"])
    parser.add_argument('--n_mfcc', type=int, default=16)
    parser.add_argument('--n_frames', type=int, default=30)
    parser.add_argument('--img_channels', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.000625)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--model_save_dir', type=str, default='models/compare')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    comparator = HeartSoundCompare(args)
    comparator.main()
    X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm, encoder = comparator.load_data()
    comparator.visualize_samples(os.path.join(args.model_save_dir, 'CNN.h5'), X_test_lstm, y_test, encoder, n=10)
