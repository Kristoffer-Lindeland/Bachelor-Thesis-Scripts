import os
import json
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

TRAIN_COVER_FOLDER = r"path/to/cover"
TRAIN_STEGO_FOLDER = r"path/to/stego"
OUTPUT_MODEL = r"path/to/models/cnn_{alg}.keras"

# used to find the correct images
PARAM_FILE = r"path/to/train.param.json"


TARGET_RATE = 0.4
TARGET_ALGORITHM = "j_uniward"  # "nsf5", "uerd", "j_uniward"
MAX_SAMPLES = 1000
BLOCK_DCT_FEATURES = 15
MAP_SIZE = (32, 32)

def zigzag_indices(n=8):
    return sorted(((x, y) for x in range(n) for y in range(n)),
                  key=lambda s: (s[0] + s[1], -s[1] if (s[0] + s[1]) % 2 else s[1]))
zigzag = zigzag_indices()

def extract_dct_map(image_path, num_coeffs=BLOCK_DCT_FEATURES, out_shape=MAP_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Make sure dimensions are multiples of 8
    h, w = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]

    blocks = []

    # Extracting DCT features, one block at the time
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = np.float32(img[i:i+8, j:j+8])
            dct = cv2.dct(block)
            coeffs = [dct[x, y] for x, y in zigzag[1:num_coeffs+1]]
            blocks.append(coeffs)

    # Stack and resize coefficient maps
    block_array = np.array(blocks).T
    dct_maps = []
    for coeff_row in block_array:
        map_1d = coeff_row.reshape(-1, 1)
        map_2d = cv2.resize(map_1d, out_shape, interpolation=cv2.INTER_LINEAR)
        dct_maps.append(map_2d)

    stacked = np.stack(dct_maps, axis=-1)
    return stacked.astype(np.float32)

def load_dataset(cover_folder, stego_folder, param_file, algorithm, rate, max_samples):
    with open(param_file, 'r') as f:
        params = json.load(f)

    # Select files that are the correct algorithm and embedding rate
    selected_filenames = [
        fname for fname, meta in params.items()
        if meta.get("rate") == rate and meta.get("steg_algorithm") == algorithm
    ][:max_samples]

    X, y = [], []

    # Extract the DCT maps from the selected images
    for fname in tqdm(selected_filenames, desc="Extracting DCT maps"):
        stego_path = os.path.join(stego_folder, fname)
        cover_path = os.path.join(cover_folder, fname)

        stego_map = extract_dct_map(stego_path)
        cover_map = extract_dct_map(cover_path)

        if stego_map is None or cover_map is None:
            continue

        X.append(stego_map)
        # Stego
        y.append(1)

        X.append(cover_map)
        # Cover
        y.append(0)

    return np.array(X), np.array(y)

def build_cnn_model(input_shape):
    # Building the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load the dataset
    X, y = load_dataset(TRAIN_COVER_FOLDER, TRAIN_STEGO_FOLDER, PARAM_FILE, TARGET_ALGORITHM, TARGET_RATE, MAX_SAMPLES)
    print(f"Loaded {len(X)} samples")

    # Splitting into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the cnn model
    model = build_cnn_model(input_shape=X.shape[1:])
    print("Training CNN.")

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=32, verbose=1)

    # Evaluate the model
    print("Evaluating model.")
    y_probs = model.predict(X_val).flatten()
    y_pred = (y_probs > 0.5).astype(int)

    print(classification_report(y_val, y_pred, digits=4))
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    # Save trained model
    model_path = OUTPUT_MODEL.format(alg=TARGET_ALGORITHM)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Save results
    results_df = pd.DataFrame({
        "index": np.arange(len(y_val)),
        "true_label": y_val,
        "predicted_label": y_pred,
        "confidence": y_probs
    })
    csv_path = model_path.replace(".keras", "_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

if __name__ == "__main__":
    main()
