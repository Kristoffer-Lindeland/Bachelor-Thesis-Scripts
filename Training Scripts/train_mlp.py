import os
import json
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

TRAIN_COVER_FOLDER = r"path/to/cover"
TRAIN_STEGO_FOLDER = r"path/to/stego"
OUTPUT_MODEL = r"path/to/models/mlp_{alg}.keras"

# used to find the correct images
PARAM_FILE = r"path/to/train.param.json"


TARGET_RATE = 0.4
TARGET_ALGORITHM = "j_uniward"  # "nsf5", "uerd", "j_uniward"
MAX_SAMPLES = 1500
NUM_COEFFS = 20

def zigzag_indices(n=8):
    return sorted(((x, y) for x in range(n) for y in range(n)),
                  key=lambda s: (s[0] + s[1], -s[1] if (s[0] + s[1]) % 2 else s[1]))

zigzag = zigzag_indices()

def extract_dct_summary_features(image_path, num_coeffs=NUM_COEFFS):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Make sure dimensions are multiples of 8
    h, w = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]

    coeff_matrix = [[] for _ in range(num_coeffs)]

    # Extracting DCT features, one block at the time
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = np.float32(img[i:i+8, j:j+8])
            dct = cv2.dct(block)
            for k in range(num_coeffs):
                x, y = zigzag[k+1]
                coeff_matrix[k].append(dct[x, y])

    # Calculate statistical features
    features = []
    for coeffs in coeff_matrix:
        coeffs = np.array(coeffs)
        features.append(np.mean(coeffs))
        features.append(np.std(coeffs))
        features.append(skew(coeffs))
        features.append(kurtosis(coeffs))

    return np.array(features)

def load_dataset(cover_folder, stego_folder, param_file, algorithm, rate, max_samples):
    with open(param_file, 'r') as f:
        parameters = json.load(f)

    selected_filenames = [
        fname for fname, meta in parameters.items()
        if meta.get("rate") == rate and meta.get("steg_algorithm") == algorithm
    ][:max_samples]

    X, y = [], []

    # Select files that are the correct algorithm and embedding rate
    for fname in tqdm(selected_filenames, desc="Extracting features"):
        stego_path = os.path.join(stego_folder, fname)
        cover_path = os.path.join(cover_folder, fname)

        if not os.path.exists(stego_path) or not os.path.exists(cover_path):
            continue

        stego_feat = extract_dct_summary_features(stego_path)
        cover_feat = extract_dct_summary_features(cover_path)

        if stego_feat is None or cover_feat is None:
            continue

        if stego_feat.shape != cover_feat.shape:
            continue

        X.append(stego_feat)
        # Stego
        y.append(1)

        X.append(cover_feat)
        # Cover
        y.append(0)

    return np.array(X), np.array(y)

# Building the model
def build_mlp_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load dataset
    X, y = load_dataset(TRAIN_COVER_FOLDER, TRAIN_STEGO_FOLDER, PARAM_FILE, TARGET_ALGORITHM, TARGET_RATE, MAX_SAMPLES)
    print(f"Loaded {len(X)} samples.")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train and test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build MLP model
    model = build_mlp_model(input_shape=(X_scaled.shape[1],))

    print("Training MLP.")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate model
    print("Evaluating.")
    y_probs = model.predict(X_val).flatten()
    y_pred = (y_probs > 0.5).astype("int32")

    print(classification_report(y_val, y_pred, digits=4, zero_division=0))
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    model_path = OUTPUT_MODEL.format(alg=TARGET_ALGORITHM)
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    # Save predictions to CSV
    results_df = pd.DataFrame({
        "index": np.arange(len(y_val)),
        "true_label": y_val,
        "predicted_label": y_pred,
        "confidence": y_probs
    })
    csv_path = model_path.replace(".keras", "_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to: {csv_path}")

if __name__ == "__main__":
    main()
