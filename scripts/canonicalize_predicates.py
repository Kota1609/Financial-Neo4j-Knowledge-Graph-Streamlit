

from __future__ import annotations
import argparse, json, os, re, multiprocessing as mp, textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ───────────────────────── config helpers ───────────────────────────

def load_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

# ───────────────────────── embeddings ───────────────────────────────

def get_embedder(model_name: str = "intfloat/e5-large-v2"):
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    return emb

# ───────────────────────── LLM wrapper ─────────────────────────────

def get_llm(model: str, temperature: float):
    from langchain_community.llms import Ollama
    return Ollama(model=model, temperature=temperature, timeout=180)

PROMPT_TMPL = textwrap.dedent("""
    You are harmonising relation names in a knowledge graph.

    Relation to canonicalise:
      phrase: {raw_rel}
      definition: {raw_def}

    Below are possible canonical relations:
    {candidates}
    F. None of the above

    Choose the SINGLE best option (A, B, C, … or F).  Reply ONLY with the
    letter.
""")

# ───────────────────────── worker process ──────────────────────────

def _worker(gpu_id: int,
            in_q: mp.Queue,
            out_q: mp.Queue,
            llm_model: str,
            temperature: float):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = get_llm(llm_model, temperature)
    while True:
        task = in_q.get()
        if task is None:
            break
        raw_rel, raw_def, candidates = task
        letters = "ABCDEFG"  # up to 6 candidates + None
        cand_txt = "\n".join(f"{letters[i]}. {rel}: {definition}" for i, (rel, definition) in enumerate(candidates))
        prompt = PROMPT_TMPL.format(raw_rel=raw_rel, raw_def=raw_def, candidates=cand_txt)
        try:
            resp = llm.invoke(prompt).strip().upper()
            choice = resp[0] if resp else "F"
        except Exception:
            choice = "F"

        if choice == "F" or choice not in letters[: len(candidates)]:
            # indicate that this should become a new canonical
            out_q.put((raw_rel, None))
        else:
            idx = ord(choice) - ord("A")
            sel_rel = candidates[idx][0]
            out_q.put((raw_rel, sel_rel))

# ───────────────────────── similarity search ───────────────────────

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ───────────────────────── orchestrator ────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDC Phase-3 canonicalisation (self)")
    ap.add_argument("--definitions", type=Path, required=True,
                    help="predicate_definitions.json from Phase-2")
    ap.add_argument("--output-dir", type=Path, default=Path("canonical_output"))
    ap.add_argument("--triplet-dir", type=Path, default=Path("triplets"))
    ap.add_argument("--embed-model", default="intfloat/e5-large-v2")
    ap.add_argument("--llm-model",   default="mixtral:8x22b-instruct")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--rewrite-triplets", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canon_path  = args.output_dir / "canonical_schema.json"
    map_path    = args.output_dir / "predicate_mapping.json"

    pred_defs: Dict[str, str] = load_json(args.definitions)
    canonical: Dict[str, str] = load_json(canon_path)
    mapping:   Dict[str, str] = load_json(map_path)

    # embedder
    embedder = get_embedder(args.embed_model)

    # cache embeddings
    canon_embs: Dict[str, np.ndarray] = {rel: embedder.encode(defn) for rel, defn in canonical.items()}

    # ---------- prepare multiprocessing ----------
    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_worker,
                        args=(gpu, in_q, out_q, args.llm_model, args.temperature),
                        daemon=True) for gpu in range(args.gpus)]
    for p in procs:
        p.start()

    # ---------- iterate over raw predicates ----------
    for raw_rel, raw_def in pred_defs.items():
        if raw_rel in mapping:
            continue  # already done

        raw_emb = embedder.encode(raw_def)
        # similarity search
        if canonical:
            sims: List[Tuple[str, float]] = [(rel, cosine(raw_emb, emb)) for rel, emb in canon_embs.items()]
            sims.sort(key=lambda x: x[1], reverse=True)
            cand_rels = sims[: args.topk]
        else:
            cand_rels = []

        # pack candidates as list[(rel, def)]
        candidates = [(rel, canonical[rel]) for rel, _ in cand_rels]
        in_q.put((raw_rel, raw_def, candidates))

    # send poison pills
    for _ in procs:
        in_q.put(None)

    processed = 0
    total = len(pred_defs)
    while processed < total:
        raw_rel, target_rel = out_q.get()
        raw_def = pred_defs[raw_rel]

        if target_rel is None:
            # create new canonical
            canonical[raw_rel] = raw_def
            canon_embs[raw_rel] = embedder.encode(raw_def)
            mapping[raw_rel] = raw_rel
        else:
            mapping[raw_rel] = target_rel
        processed += 1
        if processed % 50 == 0:
            print(f"✔ {processed}/{total} predicates processed")
            save_json(canon_path, canonical)
            save_json(map_path, mapping)

    for p in procs:
        p.join()

    save_json(canon_path, canonical)
    save_json(map_path, mapping)
    print(f"✅ Canonicalisation done → {len(canonical)} canonical relations")

    # optional: rewrite triplets
    if args.rewrite_triplets:
        out_dir = args.output_dir / "canonical_triplets"
        out_dir.mkdir(exist_ok=True)
        for fp in args.triplet_dir.glob("*_triplets_clean.json"):
            data = json.loads(fp.read_text())
            for t in data:
                t["predicate"] = mapping.get(t["predicate"], t["predicate"])
            (out_dir / fp.name.replace("_clean", "_canonical")).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"✂️  Triplets rewritten to {out_dir}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import torch  # needed for cuda check in embedder
    main() 