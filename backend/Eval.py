import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "evaluation_results.csv") # make sure this path points to your file
df = pd.read_csv(csv_path)

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

# Vizualizations

# CM
cm = confusion_matrix(y_true, y_pred, labels=df['expected_category'].unique())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=df['expected_category'].unique(),
            yticklabels=df['expected_category'].unique(), cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual Category")
plt.xlabel("Predicted Category")
plt.tight_layout()
plt.show()

# BC
plt.figure(figsize=(10, 6))
sns.barplot(x=category_hit_rates.index, y=category_hit_rates.values)
plt.xticks(rotation=45)
plt.ylabel("Hit Rate")
plt.title("Hit Rate per Category")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Histo
plt.figure(figsize=(8, 5))
sns.histplot(df['overall_conf'], bins=20, kde=True)
plt.title("Distribution of Overall Confidence Scores")
plt.xlabel("Overall Confidence")
plt.ylabel("Count")
plt.tight_layout()
plt.show()