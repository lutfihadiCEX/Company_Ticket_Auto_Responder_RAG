import os
import re

# Load the full text file
with open(r"C:\MLCourse\ticket-auto-responder\backend\raw_kb_articles.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create output folder
output_dir = os.path.join(os.path.dirname(__file__), "kb_articles")
os.makedirs(output_dir, exist_ok=True)

# Split at each "Article XX:" line
articles = re.split(r"(Article\s+\d+:\s+Title:)", text)

# Reconstruct article groups
final_articles = []
i = 1
while i < len(articles):
    title_header = articles[i]
    content = articles[i+1]
    final_articles.append(title_header + content)
    i += 2

# Function to clean filenames
def clean_filename(s):
    # Remove invalid characters
    s = re.sub(r'[<>:"/\\|?*]', '', s)
    # Remove quotes
    s = s.replace('"', '').replace("'", '')
    # Replace spaces with underscore
    s = re.sub(r'\s+', '_', s)
    # Replace multiple underscores with a single underscore
    s = re.sub(r'_+', '_', s)
    # Remove leading/trailing underscores
    s = s.strip('_')
    # Optional: lowercase
    s = s.lower()
    return s

# Save each article into its own TXT file
for idx, article in enumerate(final_articles, start=1):
    # Extract the title for filename
    title_match = re.search(r"Title:\s*(.+?)\s*Content:", article)
    if title_match:
        title = title_match.group(1)
    else:
        title = f"Article_{idx}"

    filename = clean_filename(title)
    filepath = os.path.join(output_dir, f"{idx:02d}_{filename}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(article.strip())

print("✅ All articles split into kb_articles/*.txt")
