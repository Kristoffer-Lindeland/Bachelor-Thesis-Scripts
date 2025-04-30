import os
import cv2
import numpy as np
import joblib
from scipy.stats import skew, kurtosis


def zigzag_indices(n=8):
    return sorted(((x, y) for x in range(n) for y in range(n)),
                  key=lambda s: (s[0] + s[1], -s[1] if (s[0] + s[1]) % 2 else s[1]))

# extracts DCT-based summary features from an image
def extract_dct_summary_features(image_path, num_coeffs=20):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]
    zigzag = zigzag_indices()
    coeff_matrix = [[] for _ in range(num_coeffs)]
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = np.float32(img[i:i+8, j:j+8])
            dct = cv2.dct(block)
            for k in range(num_coeffs):
                x, y = zigzag[k+1]
                coeff_matrix[k].append(dct[x, y])
    features = []
    for coeffs in coeff_matrix:
        coeffs = np.array(coeffs)
        features.extend([np.mean(coeffs), np.std(coeffs), skew(coeffs), kurtosis(coeffs)])
    return np.array(features)

# uses trained Random Forest models to predict if an image is cover or stego
def predict_rf(image_path, algorithms, rate, model_dir):
    features = extract_dct_summary_features(image_path)
    if features is None:
        return []

    results = []
    best_pred = 0
    best_conf = 0
    best_model_id = ""

    for alg in algorithms:
        model_path = os.path.join(model_dir, f"model_rf_{alg}_{rate}.joblib")
        if not os.path.exists(model_path):
            continue
        try:
            scaler, model = joblib.load(model_path)
        except Exception:
            continue

        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0]
        pred = model.predict(features_scaled)[0]
        conf = proba[1] if pred == 1 else proba[0]

        if pred == 1 and conf > 0.6:
            best_pred = pred
            best_conf = conf
            best_model_id = f"{alg}_{rate}"
            break
        elif conf > best_conf:
            best_pred = pred
            best_conf = conf
            best_model_id = f"{alg}_{rate}"

    label = "Stego" if best_pred == 1 else "Cover"
    return [(image_path, label, best_conf, best_model_id)]
