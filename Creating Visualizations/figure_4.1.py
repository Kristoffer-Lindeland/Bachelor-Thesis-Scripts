import pandas as pd
import matplotlib.pyplot as plt

counts_df = pd.DataFrame({
    'Model': ['CNN', 'SVM', 'RF', 'MLP'],
    'Correct Predictions': [1180, 1227, 1597, 1162],
    'Incorrect Predictions': [1220, 1173, 803, 1238]
})

models = counts_df['Model']
correct = counts_df['Correct Predictions']
incorrect = counts_df['Incorrect Predictions']

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(models))

bars1 = plt.bar(x, correct, width=bar_width, label='Correct', color='steelblue', align='center')
bars2 = plt.bar([i + bar_width for i in x], incorrect, width=bar_width, label='Incorrect', color='salmon', align='center')

for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 20, f'{height}',
             ha='center', va='bottom', fontsize=10)

plt.xlabel('Machine Learning Model')
plt.ylabel('Number of Predictions')
plt.title('Correct vs Incorrect Predictions by ML Model')
plt.xticks([i + bar_width / 2 for i in x], models)
plt.legend()
plt.tight_layout()

plt.show()