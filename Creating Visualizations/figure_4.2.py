import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Stego\n Predicted\n Stego\n (True Positive)': [1092, 1101, 973, 1090],
    'Cover\n Predicted\n Stego\n (False Positive)': [1112, 1074, 576, 1128],
    'Stego\n Predicted\n Cover\n (False Negative)': [88, 126, 624, 72],
    'Cover\n Predicted\n Cover\n (True Negative)': [108, 99, 227, 110]
}

df = pd.DataFrame(data, index=['CNN', 'SVM', 'RF', 'MLP'])

plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True)

plt.title('Confusion Matrix Model Predictions')
plt.xlabel('Prediction Outcome')
plt.ylabel('ML Model')

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()
