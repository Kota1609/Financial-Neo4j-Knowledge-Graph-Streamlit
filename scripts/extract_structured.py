#!/usr/bin/env python3
import os
import re
import json
import fitz  # PyMuPDF

# — automatically detect script’s folder —
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# now point input/output at the local folders
INPUT_DIR  = os.path.join(BASE_DIR, "raw_pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "structured_text")

# Regex to catch headers like "Item 1. Business" or "Item 1A. Risk Factors"
SECTION_RE = re.compile(r'^(Item\s+\d+[A-Za-z]?\.\s+.*)', re.IGNORECASE)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(INPUT_DIR, fname)
        doc = fitz.open(pdf_path)
        structured = []
        current_section = None

        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text")
            # update section if we see a header
            for line in text.splitlines():
                m = SECTION_RE.match(line.strip())
                if m:
                    current_section = m.group(1).strip()
            structured.append({
                "page":    i + 1,
                "section": current_section,
                "text":    text
            })

        out_fname = f"{os.path.splitext(fname)[0]}_structured.json"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)

        print(f"[✓] {fname} → {out_path}")

if __name__ == "__main__":
    main()
