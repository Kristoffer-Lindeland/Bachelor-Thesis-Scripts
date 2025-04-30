import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

TEST_FOLDER = r"C:\path\to\test\folder"
TRUE_LABELS_CSV = r"C:\path\to\True_labels.csv"
MODEL_PATH = r"C:\path\to\model\mlp_{}.keras"
NUM_COEFFS = 20
OUTPUT_CSV = r"C:\path\to\MLP_test_predictions.csv"

def zigzag_indices(n=8):
    return sorted(((x, y) for x in range(n) for y in range(n)),
                  key=lambda s: (s[0] + s[1], -s[1] if (s[0] + s[1]) % 2 else s[1]))

zigzag = zigzag_indices()

def extract_dct_summary_features(image_path, num_coeffs=NUM_COEFFS):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Make sure image dimensions are multiples of 8
    h, w = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w]

    coeff_matrix = [[] for _ in range(num_coeffs)]
    # Divide image into 8x8 blocks and extract DCT coefficients in zigzag order
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
        features.append(np.mean(coeffs))
        features.append(np.std(coeffs))
        features.append(skew(coeffs))
        features.append(kurtosis(coeffs))

    return np.array(features)

def main():
    # Load true labels and algorithm types
    true_df = pd.read_csv(TRUE_LABELS_CSV)
    true_df["filename"] = true_df["filename"].apply(lambda x: os.path.basename(x))
    true_labels = dict(zip(true_df["filename"], true_df["true_label"]))
    true_algorithms = dict(zip(true_df["filename"], true_df["true_algorithm"]))

    # Load models
    algorithms = ["j_uniward", "nsf5", "uerd"]
    models = {}

    for alg in algorithms:
        print(f"Loading MLP model for {alg}...")
        model = tf.keras.models.load_model(MODEL_PATH.format(alg))
        models[alg] = model

    results = []
    test_filenames = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(".jpg")]

    # Process each test image
    for fname in tqdm(test_filenames, desc="Processing test images"):
        path = os.path.join(TEST_FOLDER, fname)
        feat = extract_dct_summary_features(path)
        if feat is None or fname not in true_labels:
            continue

        model_preds = []
        model_confidences = []

        # Run image through each model and collect predictions
        for model_alg in algorithms:
            model = models[model_alg]
            x_input = np.expand_dims(feat, axis=0)
            proba = model.predict(x_input, verbose=0)[0][0]
            model_preds.append(proba >= 0.5)
            model_confidences.append(proba)

        # Voting algorithm
        final_prediction = 1 if any(model_preds) else 0
        max_confidence = max(model_confidences)

        # Save prediction result for image
        results.append({
            "filename": fname,
            "true_label": true_labels[fname],
            "true_algorithm": true_algorithms.get(fname, "unknown"),
            "predicted_label": final_prediction,
            "confidence": max_confidence
        })

    # Evaluate predictions
    df = pd.DataFrame(results)
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    print("\n--- Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved prediction results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
