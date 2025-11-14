# scripts/download_arxiv.py
import requests
from xml.etree import ElementTree as ET
from tqdm import tqdm
import os

OUTPUT_DIR = "data/raw_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def query_arxiv(search_query="cat:cs.CL", max_results=50):
    url = "http://export.arxiv.o../app/query"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def download_pdfs(xml_text):
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    for e in tqdm(entries):
        pdf_link = None
        title = e.find("atom:title", ns).text.strip()
        id_ = e.find("atom:id", ns).text.split("/")[-1]
        for link in e.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib["href"]
        if not pdf_link:
            pdf_link = f"https://arxiv.org/pdf/{id_}.pdf"
        filename = os.path.join(OUTPUT_DIR, f"{id_}.pdf")
        if os.path.exists(filename):
            continue
        try:
            with requests.get(pdf_link, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(1024*32):
                        f.write(chunk)
        except Exception as ex:
            print("failed", pdf_link, ex)

if __name__ == "__main__":
    xml_text = query_arxiv("cat:cs.CL", max_results=50)
    download_pdfs(xml_text)
