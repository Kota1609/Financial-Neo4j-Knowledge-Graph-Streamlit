
import asyncio, json, os, re, sys, textwrap, multiprocessing as mp
from pathlib import Path
from typing  import List, Dict, Any

# ───────────────────────── locate folders ──────────────────────────
HERE       = Path(__file__).resolve()
PDF_ROOT   = next(p for p in HERE.parents if (p / "chunks").exists())
CHUNK_FILE = PDF_ROOT / "chunks"   / "Apple_chunks.json"
OUT_DIR    = PDF_ROOT / "triplets"
OUT_DIR.mkdir(exist_ok=True)
RAW_OUT    = OUT_DIR  / "Apple_triplets.jsonl"   # streamed
FINAL_OUT  = OUT_DIR  / "Apple_triplets.json"    # pretty

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

# very permissive pattern → first []-block = JSON we want
JSON_RE = re.compile(r"\[[\s\S]*?\]", re.MULTILINE)

def _repair(raw: str) -> List[Dict[str, str]]:
    """
    1. pick the first [] block
    2. try json.loads → if it fails, apply a few heuristics
    3. guarantee each triple has 3 string fields
    """
    m = JSON_RE.search(raw)
    if not m:
        return []

    snippet = m.group(0)

    # small heuristics: trailing commas, single quotes, line comments
    cleaned = re.sub(r",\s*]", "]", snippet)
    cleaned = cleaned.replace("'", '"')

    try:
        triples = json.loads(cleaned)
    except Exception:
        return []

    good: List[Dict[str, str]] = []
    for t in triples:
        if not isinstance(t, dict):                 # skip weird items
            continue
        s = str(t.get("subject", "")).strip()[:500]
        p = str(t.get("predicate", "")).strip()[:500]
        o = str(t.get("object", "")).strip()[:500]
        if s and p and o:
            good.append({"subject": s, "predicate": p, "object": o})
    return good

# ───────────────────────── worker process  ─────────────────────────
def _worker(gpu_id: int, in_q: mp.Queue, out_q: mp.Queue):
    """
    Each worker:
      • sets CUDA_VISIBLE_DEVICES to *one* GPU
      • loads mixtral once
      • processes requests forever
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = Ollama(
        model      = MODEL,
        temperature= TEMPERATURE,
        timeout    = TIMEOUT,
        callbacks  = [StreamingStdOutCallbackHandler()],
    )

    while True:
        task = in_q.get()
        if task is None:        # poison pill → exit
            break
        idx, chunk = task
        prompt = PROMPT_TMPL.format(
            chunk_id = chunk["chunk_id"],
            page     = chunk["page_number"],
            text     = chunk["text"],
        )

        try:
            resp     = llm(prompt)
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

    # –– spin up 8 GPU workers –––––––––––––––––––
    in_q, out_q = mp.Queue(), mp.Queue()
    procs = [
        mp.Process(target=_worker, args=(gpu, in_q, out_q), daemon=True)
        for gpu in range(8)
    ]
    for p in procs:
        p.start()

    # –– async fan-out: feed queues –––––––––––––––
    total      = len(chunks)
    next_send  = 0
    next_recv  = 0
    pending    = {}
    sem        = asyncio.Semaphore(max_conc)

    RAW_OUT.unlink(missing_ok=True)   # start fresh
    async with aiofiles.open(RAW_OUT, "w") as raw_f:   # type: ignore
        while next_recv < total:
            # fill pipeline
            while next_send < total and len(pending) < max_conc:
                await sem.acquire()
                in_q.put((next_send, chunks[next_send]))
                pending[next_send] = chunks[next_send]
                next_send += 1

            # wait for one finished item
            idx, triples, status = await asyncio.get_running_loop().run_in_executor(
                None, out_q.get
            )
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

            print(f"{next_recv:>5}/{total}  {chunk['chunk_id']:25} – "
                  f"{len(triples)} triples  ({status})")

    # kill workers
    for _ in procs:
        in_q.put(None)
    for p in procs:
        p.join()

    # –– combine jsonl → pretty JSON array –––––––––
    with open(RAW_OUT, "r", encoding="utf-8") as r:
        all_records = [json.loads(l) for l in r]
    FINAL_OUT.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\n✅  wrote {RAW_OUT.name}  and  {FINAL_OUT.name}")

# ───────────────────────── entry-point  ────────────────────────────
if __name__ == "__main__":
    import argparse, aiofiles       # aiofiles only used in main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_CONC,
                        help="total async requests in flight (default 64)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.max_concurrency))
    except KeyboardInterrupt:
        sys.exit(1)
