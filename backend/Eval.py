import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("evaluation_results.csv")  # make sure this path points to your file

overall_accuracy = df['hit'].mean()
print(f"Overall Hit Accuracy: {overall_accuracy:.4f}\n")

y_true = df['expected_category']
y_pred = df['predicted_category']

report = classification_report(y_true, y_pred, zero_division=0)
print("Classification Report (Precision, Recall, F1-score per category):\n")
print(report)

category_hit_rates = df.groupby('expected_category')['hit'].mean()
print("\nHit rate per category:")
print(category_hit_rates)