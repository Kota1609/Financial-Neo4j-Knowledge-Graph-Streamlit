
import json, re, unicodedata, hashlib
from pathlib import Path
from typing import Dict, List

ROOT      = Path(__file__).resolve().parent         #  ← fixed
RAW_FILE  = ROOT / "triplets" / "Intel_triplets.json"
OUT_FILE  = ROOT / "triplets" / "Intel_triplets_clean.json"

_WS_RE  = re.compile(r"\s+")
PUNCT   = str.maketrans("", "", r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~""")

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = text.translate(PUNCT).lower()
    text = _WS_RE.sub(" ", text).strip()
    return text

def dedup_key(s: str, p: str, o: str) -> str:
    return hashlib.md5(f"{s}<<{p}<<{o}".encode()).hexdigest()

def cleanse(records: List[Dict]) -> List[Dict]:
    seen, cleaned = set(), []
    for rec in records:
        for t in rec["triplets"]:
            s_raw, p_raw, o_raw = t["subject"], t["predicate"], t["object"]
            if not (s_raw and p_raw and o_raw):
                continue
            key = dedup_key(normalise(s_raw), normalise(p_raw), normalise(o_raw))
            if key in seen:
                [c for c in cleaned if c["_key"] == key][0]["prov"].append(rec["chunk_id"])
                continue
            cleaned.append({
                "_key": key,
                "subject":  s_raw.strip(),
                "predicate":p_raw.strip(),
                "object":   o_raw.strip(),
                "prov":     [rec["chunk_id"]],
                "page":     rec["page_number"],
            })
            seen.add(key)
    return cleaned

def main():
    records  = json.load(open(RAW_FILE, encoding="utf-8"))
    triplets = cleanse(records)
    json.dump(triplets, open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅  wrote {len(triplets):,} clean triples → {OUT_FILE.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
