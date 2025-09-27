import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
import pywt
import random
import argparse


class HeartSoundClassification:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.model_save_path = args.model_save_path
        self.labels = args.labels
        self.n_mfcc = args.n_mfcc
        self.n_frames = args.n_frames
        self.img_channels = args.img_channels
        self.acc_plot_path = args.acc_plot_path
        self.loss_plot_path = args.loss_plot_path
        self.test_size = args.test_size
        self.validation_split = args.validation_split
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate

    def load_data(self):
        features = []
        file_labels = []
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

                        # Stack MFCC + DWT as channels
                        combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)
                        features.append(combined_feat)
                        file_labels.append(label)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
        X = np.array(features)
        y = np.array(file_labels)

        # One-hot encode labels
        encoder = LabelBinarizer()
        y_onehot = encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=self.test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test, encoder

    def build_model(self, input_shape):
        try:
            print("Building model...")
            model = Sequential([
                layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
                BatchNormalization(),
                layers.MaxPooling2D((2,2)),
                Dropout(self.dropout_rate),
                
                layers.Conv2D(64, (3,3), activation='relu'),
                BatchNormalization(),
                layers.MaxPooling2D((2,2)),
                Dropout(self.dropout_rate),
                
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate/2),
                Dense(len(self.labels), activation='softmax')
            ])
            
            # Compile the model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.summary()
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            raise

    def train_model(self, model, X_train, y_train, X_test, y_test):
        if not os.path.exists(self.model_save_path):
            try:
                print("Training model...")
                # Define callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # Save the model
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                model.save(self.model_save_path)
                print(f"Model saved at {self.model_save_path}")

                return history, model
            except Exception as e:
                print(f"Error training model: {e}")
                raise
        else:
            print(f"Model already exists at {self.model_save_path}")
            model = models.load_model(self.model_save_path)
            return None, model

    def evaluate_model(self, model, X_test, y_test):
        try:
            print("Evaluating model...")
            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Predictions
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            raise

    def plot_training_history(self, history):
        if history is None:
            # Display saved plots
            for plot_path in [self.acc_plot_path, self.loss_plot_path]:
                if os.path.exists(plot_path):
                    img = plt.imread(plot_path)
                    plt.figure(figsize=(8,5))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
        else:
            try:
                print("Plotting training history...")
                # Accuracy plot
                plt.figure(figsize=(8,5))
                plt.plot(history.history['accuracy'], label='Train Acc')
                plt.plot(history.history['val_accuracy'], label='Val Acc')
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.ylim(0, 1)
                plt.legend()
                plt.title("Training vs Validation Accuracy")
                plt.tight_layout()
                plt.savefig(self.acc_plot_path)
                print(f"Accuracy plot saved at {self.acc_plot_path}")
                plt.show()

                # Loss plot
                plt.figure(figsize=(8,5))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.ylim(0, 1)
                plt.legend()
                plt.title("Training vs Validation Loss")
                plt.tight_layout()
                plt.savefig(self.loss_plot_path)
                print(f"Loss plot saved at {self.loss_plot_path}")
                plt.show()
            except Exception as e:
                print(f"Error plotting training history: {e}")
                raise

    def test_random_samples(self, n=10, encoder=None):
        model = models.load_model(self.model_save_path)

        # Collect all test files
        test_files = []
        for label in self.labels:
            folder = os.path.join(self.data_path, label)
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    test_files.append((os.path.join(folder, file), label))

        random_files = random.sample(test_files, n)

        plt.figure(figsize=(15, 10))
        for idx, (filepath, true_label) in enumerate(random_files, 1):
            y_audio, sr = librosa.load(filepath, sr=None)

            # MFCC + DWT for prediction
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

            combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)
            combined_feat = combined_feat[np.newaxis, ...]  # add batch dim

            pred_prob = model.predict(combined_feat)
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
        # Load and preprocess data
        X_train, X_test, y_train, y_test, encoder = self.load_data()
        
        # Build and train model
        model = self.build_model((self.n_mfcc, self.n_frames, self.img_channels))
        history, model = self.train_model(model, X_train, y_train, X_test, y_test)
        
        # Evaluate model
        self.evaluate_model(model, X_test, y_test)
        
        # Plot training history
        self.plot_training_history(history)
        self.test_random_samples(10, encoder)

        return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Heart Sound Classification')
    parser.add_argument('--data_path', type=str, default="D:/Coding/NCKH/Classification-Of-Heart-Sound-Signal-Using-Multiple-Features/Data",
                       help='Path to the dataset')
    parser.add_argument('--model_save_path', type=str, default='models/heart_model.h5',
                       help='Path to save the trained model')
    parser.add_argument('--labels', type=list, default=["MR", "MS", "N", "MVP"],
                       help='Labels for classification')
    parser.add_argument('--n_mfcc', type=int, default=16, 
                        help='Number of MFCC features to extract')
    parser.add_argument('--n_frames', type=int, default=30,
                        help='Number of frames for each MFCC feature')
    parser.add_argument('--img_channels', type=int, default=2,
                        help='Number of channels in the input image (e.g., MFCC + DWT)')
    parser.add_argument('--acc_plot_path', type=str, default='models/accuracy_plot.png',
                       help='Path to save the accuracy plot')
    parser.add_argument('--loss_plot_path', type=str, default='models/loss_plot.png',
                       help='Path to save the loss plot')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Proportion of training data to use for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.000625,
                       help='Learning rate for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    classifier = HeartSoundClassification(args)
    classifier.main()