import os
import json
import requests
from bs4 import BeautifulSoup
import trafilatura
from PIL import Image
import pytesseract

# -----------------------------
# Configuration
# -----------------------------
BASE_URL = "https://arxiv.org/list/cs.CL/recent"
OUTPUT_FILE = "arxiv_clean.json"
SCREENSHOT_DIR = "screenshots"  # if you have screenshots for OCR

# Ensure screenshot directory exists
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# -----------------------------
# 1. Fetch arXiv listing page
# -----------------------------
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, "html.parser")

# Get paper URLs
paper_links = []
for a in soup.find_all("a", title="Abstract"):
    href = a.get("href")
    if href.startswith("/abs/"):
        paper_links.append("https://arxiv.org" + href)

# Limit to latest 200 papers
paper_links = paper_links[:200]

# -----------------------------
# 2. Scrape each paper
# -----------------------------
papers_data = []

for url in paper_links:
    r = requests.get(url)
    html = r.text

    # Extract title
    soup_paper = BeautifulSoup(html, "html.parser")
    title_tag = soup_paper.find("h1", class_="title")
    title = title_tag.text.replace("Title:", "").strip() if title_tag else ""

    # Extract authors
    authors_tag = soup_paper.find("div", class_="authors")
    authors = authors_tag.text.replace("Authors:", "").strip() if authors_tag else ""

    # Extract date
    date_tag = soup_paper.find("div", class_="dateline")
    date = date_tag.text.strip() if date_tag else ""

    # -----------------------------
    # 3a. Clean HTML abstract using Trafilatura
    # -----------------------------
    abstract_tag = soup_paper.find("blockquote", class_="abstract")
    abstract_html = str(abstract_tag) if abstract_tag else ""
    abstract_clean = trafilatura.extract(abstract_html, include_comments=False, include_tables=False) or ""

    # -----------------------------
    # 3b. Optional: OCR from screenshots
    # -----------------------------
    screenshot_path = os.path.join(SCREENSHOT_DIR, url.split("/")[-1] + ".png")
    if os.path.exists(screenshot_path):
        try:
            img = Image.open(screenshot_path)
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                abstract_clean += "\n" + ocr_text.strip()
        except Exception as e:
            print(f"OCR failed for {url}: {e}")

    # -----------------------------
    # 4. Append data
    # -----------------------------
    papers_data.append({
        "url": url,
        "title": title,
        "abstract": abstract_clean,
        "authors": authors,
        "date": date
    })

# -----------------------------
# 5. Save to JSON
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(papers_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(papers_data)} papers to {OUTPUT_FILE}")
