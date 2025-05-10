#!/usr/bin/env python3
"""
Apple 10-K → SPO-triplet extractor  (multi-GPU, fault-tolerant)

• spins up 8 worker-processes (one per GPU) that each host
  `mixtral:8x22b-instruct` via Ollama
• fans out async requests to all workers (default 64 in-flight)
• repairs / validates the JSON the model returns
• streams incremental results to  PDF/triplets/Apple_triplets.jsonl
  (one line = one chunk with its triplets)
• at the end it also writes the final pretty-printed JSON array that your
  downstream KG code expects.
"""

import asyncio, json, os, re, sys, textwrap, multiprocessing as mp
from pathlib import Path
from typing  import List, Dict, Any

# ───────────────────────── locate folders ──────────────────────────
HERE       = Path(__file__).resolve()
PDF_ROOT   = next(p for p in HERE.parents if (p / "chunks").exists())
CHUNK_FILE = PDF_ROOT / "chunks"   / "Alphabet-Google_chunks.json"
OUT_DIR    = PDF_ROOT / "triplets"
OUT_DIR.mkdir(exist_ok=True)
RAW_OUT    = OUT_DIR  / "Google_triplets.jsonl"   # streamed
FINAL_OUT  = OUT_DIR  / "Google_triplets.json"    # pretty

# ───────────────────────── Ollama wrapper  ─────────────────────────
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL          = "mixtral:8x22b-instruct"
TEMPERATURE    = 0.05
TIMEOUT        = 180          # seconds per request
DEFAULT_CONC   = 64           # total in-flight requests (across all GPUs)

PROMPT_TMPL = textwrap.dedent("""\
    SYSTEM:
    You are an information-extraction engine.  
    You ONLY respond with valid JSON — no prose.

    USER:
    From the paragraph below, extract all *factual*
    (subject, predicate, object) triples.  
    Return them **exactly** as a JSON array, where each
    element is an object with keys "subject", "predicate", "object".
    If you cannot find any factual relations, return [].

    ---
    chunk_id: {chunk_id}
    page:     {page}

    TEXT:
    {text}
""")

# ––––– find first JSON array or object that contains "triples" array –––
JSON_RE = re.compile(r"\[[\s\S]*?\]", re.MULTILINE)
OBJ_RE  = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)

def _repair(raw: str) -> List[Dict[str, str]]:
    """Return a cleaned list of {subject,predicate,object} dicts."""
    snippet: str | None = None

    m = JSON_RE.search(raw)
    if m:
        snippet = m.group(0)
    else:
        obj = OBJ_RE.search(raw)
        if obj:
            snippet = obj.group(0)

    if not snippet:
        return []

    cleaned = re.sub(r",\s*]", "]", snippet)  # trailing commas
    cleaned = cleaned.replace("'", '"')          # single → double quotes

    try:
        data = json.loads(cleaned)
    except Exception:
        return []

    triples = data if isinstance(data, list) else data.get("triples", [])

    good: List[Dict[str, str]] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        s = str(t.get("subject", "")).strip()[:500]
        p = str(t.get("predicate", "")).strip()[:500]
        o = str(t.get("object", "")).strip()[:500]
        if s and p and o:
            good.append({"subject": s, "predicate": p, "object": o})
    return good

# ───────────────────────── worker process  ─────────────────────────

def _worker(gpu_id: int, in_q: mp.Queue, out_q: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = Ollama(
        model       = MODEL,
        temperature = TEMPERATURE,
        timeout     = TIMEOUT,
        callbacks   = [StreamingStdOutCallbackHandler()],
    )

    while True:
        task = in_q.get()
        if task is None:
            break  # poison pill
        idx, chunk = task
        prompt = PROMPT_TMPL.format(
            chunk_id = chunk["chunk_id"],
            page     = chunk["page_number"],
            text     = chunk["text"],
        )
        try:
            resp     = llm.invoke(prompt)  # modern API
            triples  = _repair(resp)
            status   = "ok"
        except Exception as e:
            triples  = []
            status   = f"error: {e}"
        out_q.put((idx, triples, status))

# ───────────────────────── orchestrator  ───────────────────────────
async def main(max_conc: int):
    chunks = json.loads(CHUNK_FILE.read_text())
    print(f"Loaded {len(chunks):,} chunks")

    in_q, out_q = mp.Queue(), mp.Queue()
    procs = [mp.Process(target=_worker, args=(gpu, in_q, out_q), daemon=True)
             for gpu in range(8)]
    for p in procs:
        p.start()

    total, next_send, next_recv = len(chunks), 0, 0
    pending: dict[int, Any] = {}
    sem = asyncio.Semaphore(max_conc)

    RAW_OUT.unlink(missing_ok=True)
    import aiofiles  # local import because only used here
    async with aiofiles.open(RAW_OUT, "w") as raw_f:  # type: ignore
        while next_recv < total:
            while next_send < total and len(pending) < max_conc:
                await sem.acquire()
                in_q.put((next_send, chunks[next_send]))
                pending[next_send] = chunks[next_send]
                next_send += 1

            idx, triples, status = await asyncio.get_running_loop().run_in_executor(None, out_q.get)
            sem.release()

            chunk = pending.pop(idx)
            record = {
                "chunk_id":      chunk["chunk_id"],
                "section_title": chunk.get("section_title", ""),
                "page_number":   chunk["page_number"],
                "text":          chunk["text"],
                "triplets":      triples,
            }
            await raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            next_recv += 1
            print(f"{next_recv:>5}/{total}  {chunk['chunk_id']:25} – {len(triples)} triples  ({status})")

    for _ in procs:
        in_q.put(None)
    for p in procs:
        p.join()

    with open(RAW_OUT, "r", encoding="utf-8") as r:
        all_records = [json.loads(l) for l in r]
    FINAL_OUT.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\n✅  wrote {RAW_OUT.name}  and  {FINAL_OUT.name}")

# ───────────────────────── entry-point  ────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONC,
                        help="total async requests in flight (default 64)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.max_concurrency))
    except KeyboardInterrupt:
        sys.exit(1) 