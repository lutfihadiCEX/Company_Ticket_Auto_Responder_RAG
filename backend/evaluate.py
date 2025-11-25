import json
from classifier import classify_ticket
from retriever import retrieve_documents
import csv
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "test_dataset.json")
output_file = os.path.join(BASE_DIR, "evaluation_results.csv")

with open(dataset_path, "r", encoding="utf-8") as f:
    test_emails = json.load(f)

results = []

for email in tqdm(test_emails):
    subject = email["subject"]
    body = email["body"]
    expected_category = email["expected_category"]

    predicted_category, classifier_conf = classify_ticket(subject, body)

    kb_docs = retrieve_documents(body, top_k=3)
    retrieval_conf = max([doc.get("similarity", 0) for doc in kb_docs], default=0)

    overall_conf = 0.7 * classifier_conf + 0.3 * retrieval_conf

    hit = 1 if predicted_category == expected_category else 0

    results.append({
        "subject": subject,
        "expected_category": expected_category,
        "predicted_category": predicted_category,
        "classifier_conf": classifier_conf,
        "retrieval_conf": retrieval_conf,
        "overall_conf": overall_conf,
        "hit": hit
    })

keys = results[0].keys()
with open(output_file, "w", newline="", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)


print(f"Evaluation complete! Results saved to {output_file}")