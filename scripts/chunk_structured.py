
import os, re, json, textwrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IN_DIR   = BASE_DIR / "structured_text"
OUT_DIR  = BASE_DIR / "chunks"

# -------- helpers ----------------------------------------------------------- #
SEC_FOOTER_RE = re.compile(r"^\s*Apple Inc\.\s*\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+\s*$", re.I)
BULLET_RE     = re.compile(r"^\s*(?:[-–•●]|[a-z]?\)|\d+\.)\s+")   # bullets / numbered items
CHECKBOX_RE   = re.compile(r"☒|☐")                               # keep “Yes ☒ No ☐” lines

MAX_CHARS     = 750         # ≈ 180 - 220 tokens (OpenAI tiktoken estimate)
OVERLAP_CHARS = 120         # ~ 30 tokens sliding window


def clean_lines(text: str) -> list[str]:
    """Strip SEC footers & empty lines, keep bullets as separate units."""
    lines: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or SEC_FOOTER_RE.match(ln):
            continue
        lines.append(ln)
    return lines


def should_keep(raw: str) -> bool:
    raw = raw.strip()
    return len(raw) >= 40 or CHECKBOX_RE.search(raw)


def to_sentences(lines: list[str]) -> list[str]:
    """
    Naïve sentence splitter good enough for SEC prose:
    split on period-space unless inside acronyms/numbers.
    """
    sents: list[str] = []
    buff: list[str] = []
    for ln in lines:
        if BULLET_RE.match(ln):
            # bullet is already a boundary
            if buff:
                sents.append(" ".join(buff))
                buff.clear()
            sents.append(ln)
            continue

        parts = re.split(r"(?<=[a-z0-9)][.?!])\s{1,}", ln)
        for p in parts:
            buff.append(p)
            if p.endswith((".", "?", "!")) and len(" ".join(buff)) > 100:
                sents.append(" ".join(buff))
                buff.clear()
    if buff:
        sents.append(" ".join(buff))
    return [s.strip() for s in sents if should_keep(s)]


def pack_sentences(sents: list[str]) -> list[str]:
    """
    Greedy packer: group sentences ≤ MAX_CHARS with OVERLAP_CHARS overlap.
    """
    chunks: list[str] = []
    buf: list[str]   = []
    for s in sents:
        trial = " ".join(buf + [s])
        if len(trial) <= MAX_CHARS:
            buf.append(s)
        else:
            if buf:
                chunks.append(" ".join(buf))
            # overlap last  overlap chunk_size
            buf = [s]  # start new
    if buf:
        chunks.append(" ".join(buf))

    # add overlaps
    final: list[str] = []
    for ch in chunks:
        if final and OVERLAP_CHARS:
            window = final[-1][-OVERLAP_CHARS:]
            ch = f"{window} {ch}"
        final.append(ch)
    return final
# ---------------------------------------------------------------------------- #


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    for fp in IN_DIR.glob("*_structured.json"):
        pdf_name  = fp.name.replace("_structured.json", "")
        out_path  = OUT_DIR / f"{pdf_name}_chunks.json"

        pages = json.loads(fp.read_text(encoding="utf-8"))

        chunks = []
        for page in pages:
            page_no  = page["page"]
            section  = page["section"] or ""
            lines    = clean_lines(page["text"])
            sents    = to_sentences(lines)
            passages = pack_sentences(sents)

            for idx, passage in enumerate(passages, 1):
                chunks.append(
                    {
                        "chunk_id"      : f"{pdf_name}_p{page_no}_c{idx}",
                        "page_number"   : page_no,
                        "section_title" : section,
                        "text"          : passage.strip()
                    }
                )

        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[✓] {pdf_name}: {len(chunks):,} clean chunks  →  {out_path}")


if __name__ == "__main__":
    main()
