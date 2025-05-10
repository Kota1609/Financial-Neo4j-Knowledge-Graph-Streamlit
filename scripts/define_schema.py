

from __future__ import annotations
import argparse, json, multiprocessing as mp, os, re, textwrap
from pathlib import Path
from typing import List, Dict

# ────────────────────────── Prompt template ───────────────────────────
_PROMPT_HEADER = textwrap.dedent("""\
    You are a financial knowledge-graph engineer.
    For EACH relation phrase below, write ONE short sentence that defines
    what the predicate means in the context of SEC Form 10-K filings.
    Use the pattern "The subject … the object …".
    Return a single JSON object whose keys are the original phrases and whose
    values are the definitions.

    Relations:
""")

def build_prompt(preds: List[str]) -> str:
    body = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preds))
    return _PROMPT_HEADER + body + "\n"

# ────────────────────────── Simple JSON repair ────────────────────────
# handles plain { .. } or fenced ```json blocks; ignores ASSISTANT:/SYSTEM: preambles
_JSON_RE = re.compile(
    r"```json\s*(\{[\s\S]*?\})\s*```"      # fenced block
    r"|\{[\s\S]*?\}",                       # or bare object
    re.MULTILINE,
)

def safe_json(text: str) -> Dict[str, str]:
    """Extract and parse the first JSON object from `text`.

    1. Strips any leading role prefixes like "ASSISTANT:"/"SYSTEM:".
    2. Accepts fenced ```json ... ``` blocks *or* bare { … } objects.
    3. Performs minimal repairs (single → double quotes, trailing commas).
    """

    # drop common role prefixes that Ollama may echo
    for prefix in ("ASSISTANT:", "SYSTEM:"):
        if prefix in text:
            text = text.split(prefix, 1)[-1]

    m = _JSON_RE.search(text)
    if not m:
        return {}

    snippet = m.group(1) or m.group(0)
    snippet = snippet.replace("'", '"')
    snippet = re.sub(r",\s*([}\]])", r"\1", snippet)  # remove trailing commas

    try:
        data = json.loads(snippet)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# ────────────────────────── Worker process ────────────────────────────

def _worker(gpu_id: int,
            in_q: mp.Queue,
            out_q: mp.Queue,
            model: str,
            temperature: float,
            timeout: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from langchain_community.llms import Ollama  # local import
    llm = Ollama(model=model,
                 temperature=temperature,
                 timeout=timeout)

    while True:
        task = in_q.get()
        if task is None:
            break
        batch_id, preds = task
        prompt = build_prompt(preds)
        try:
            resp = llm.invoke(prompt)
            mapping = safe_json(resp)
            if not mapping:
                # fallback attempt per predicate
                mapping = {}
                for p in preds:
                    single = llm.invoke(build_prompt([p]))
                    m = safe_json(single)
                    mapping[p] = m.get(p, "")
        except Exception as e:
            mapping = {p: f"ERROR: {e}" for p in preds}
        out_q.put((batch_id, mapping))

# ────────────────────────── Predicate collection ───────────────────────

def collect_predicates(triplet_dir: Path) -> List[str]:
    preds = set()
    for fp in triplet_dir.glob("*_triplets_clean.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            print(f"⚠️  Could not parse {fp.name}: {e}")
            continue
        for t in data:
            p = str(t.get("predicate", "")).strip()
            if p:
                preds.add(re.sub(r"\s+", " ", p))
    return sorted(preds)

# ────────────────────────── Main orchestrator ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDC Phase-2 – predicate definition (multi-GPU)")
    ap.add_argument("--triplet-dir", type=Path, default=Path("triplets"))
    ap.add_argument("--output-dir", type=Path,  default=Path("schema_definitions"))
    ap.add_argument("--model", default="mixtral:8x22b-instruct")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int,   default=20)
    ap.add_argument("--gpus", type=int, default=8,
                    help="number of GPUs / worker procs to spawn")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "predicate_definitions.json"
    csv_path  = args.output_dir / "predicate_definitions.csv"

    # ---------- load / collect ----------
    all_preds = collect_predicates(args.triplet_dir)
    print(f"🔍 Found {len(all_preds):,} unique predicates in {args.triplet_dir}/")

    defined: Dict[str, str] = {}
    if json_path.exists():
        defined = json.loads(json_path.read_text())
        print(f"↪️  {len(defined)} predicates already defined – skipping them")

    todo = [p for p in all_preds if p not in defined]
    if not todo:
        print("✅ Nothing to do – dataset already fully defined.")
        return
    print(f"✏️  Need to define {len(todo)} new predicates")

    # ---------- batching ----------
    batches: List[List[str]] = [todo[i:i+args.batch_size] for i in range(0, len(todo), args.batch_size)]
    total_batches = len(batches)

    # ---------- multiprocessing ----------
    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_worker,
                        args=(gpu, in_q, out_q,
                              args.model, args.temperature, args.timeout),
                        daemon=True)
             for gpu in range(args.gpus)]
    for p in procs:
        p.start()

    for idx, batch in enumerate(batches):
        in_q.put((idx, batch))

    # send poison pills
    for _ in procs:
        in_q.put(None)

    processed = 0
    while processed < total_batches:
        batch_id, mapping = out_q.get()
        defined.update({k: v for k, v in mapping.items() if v})
        processed += 1
        print(f"📦  Batch {batch_id+1}/{total_batches} done – total defs {len(defined)}")

        # incremental save
        json_path.write_text(json.dumps(defined, indent=2, ensure_ascii=False))
        with csv_path.open("w", encoding="utf-8") as f:
            for k, v in sorted(defined.items()):
                f.write(f'"{k.replace("\"", "''")}","{v.replace("\"", "''")}"\n')

    for p in procs:
        p.join()

    print(f"✅ Phase-2 complete – {len(defined)} predicates defined.")
    print(f"   JSON → {json_path}\n   CSV  → {csv_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 