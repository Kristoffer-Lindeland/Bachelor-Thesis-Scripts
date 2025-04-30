import os
import json
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

TRAIN_COVER_FOLDER = r"path/to/cover"
TRAIN_STEGO_FOLDER = r"path/to/stego"
OUTPUT_MODEL = r"path/to/models/svm_{alg}.joblib"

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

def extract_dct_map_flat(image_path, num_coeffs=BLOCK_DCT_FEATURES, out_shape=MAP_SIZE):
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

    block_array = np.array(blocks).T
    dct_maps = []
    for coeff_row in block_array:
        map_1d = coeff_row.reshape(-1, 1)
        map_2d = cv2.resize(map_1d, out_shape, interpolation=cv2.INTER_LINEAR)
        dct_maps.append(map_2d)

    stacked = np.stack(dct_maps, axis=-1)  # (H, W, C)
    return stacked.flatten().astype(np.float32)

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
    for fname in tqdm(selected_filenames, desc="Extracting DCT features (SVM)"):
        stego_path = os.path.join(stego_folder, fname)
        cover_path = os.path.join(cover_folder, fname)

        stego_feat = extract_dct_map_flat(stego_path)
        cover_feat = extract_dct_map_flat(cover_path)

        if stego_feat is None or cover_feat is None:
            continue

        X.append(stego_feat)
        # Stego
        y.append(1)

        X.append(cover_feat)
        # Cover
        y.append(0)

    return np.array(X), np.array(y)

def main():
    X, y = load_dataset(TRAIN_COVER_FOLDER, TRAIN_STEGO_FOLDER, PARAM_FILE, TARGET_ALGORITHM, TARGET_RATE, MAX_SAMPLES)
    print(f"Loaded {len(X)} samples")

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train and test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Training the model
    print("Training SVM.")
    svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
    calibrated_svm = CalibratedClassifierCV(estimator=svm, method='sigmoid', cv=5)
    calibrated_svm.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating.")
    y_pred = calibrated_svm.predict(X_val)
    print(classification_report(y_val, y_pred, digits=4))
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    # Save model and scaler
    model_path = OUTPUT_MODEL.format(alg=TARGET_ALGORITHM)
    print(f"Saving SVM model to: {model_path}")
    joblib.dump((scaler, calibrated_svm), model_path)

if __name__ == "__main__":
    main()
