
from __future__ import annotations
import argparse, itertools, json, os, re, hashlib
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

################################################################################
# ----------------------------- configuration ----------------------------------
################################################################################

def load_env():
    """Search upwards for a .env and load it."""
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env", Path(".env")]:
        if p.exists():
            load_dotenv(p)
            break

load_env()

################################################################################
# ----------------------------- helpers ---------------------------------------
################################################################################

def normalize(text: str | None, company: str = "", context: str = "") -> str:
    """Create a normalized key for an entity, ensuring uniqueness."""
    if not text:
        return "none"
    
    # Include context in the hash for better differentiation
    hash_input = f"{company}:{text}:{context}"
    text_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Special handling for monetary values
    if text.startswith('$') or re.search(r'^\$?[\d,.]+\s?(million|billion|trillion|thousand)?$', text, re.IGNORECASE):
        # For monetary values, create a company+context specific key
        normalized = f"{re.sub(r'[^\w]', '', text.lower())}_{company}_{text_hash}"
    else:
        # Regular normalization for other text with company context
        base = re.sub(r"[^0-9A-Za-z]+", "_", text.lower()).strip("_")
        normalized = f"{base}_{company}_{text_hash}"
    
    # truncate to 120 chars to keep Neo4j happy
    return normalized[:120]


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            break
        yield batch

################################################################################
# ----------------------------- Cypher ----------------------------------------
################################################################################

CREATE_KEY_CONSTRAINT = """
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE;
"""

UPSERT_NODE = """
MERGE (e:Entity {key:$key})
  ON CREATE SET e.name = $name, e.company = $company
RETURN e
"""

UPSERT_REL = """
MATCH (s:Entity {key:$s_key})
MATCH (o:Entity {key:$o_key})
MERGE (s)-[r:FACT {predicate:$pred, company:$company}]->(o)
  ON CREATE SET r.page = $page
  ON MATCH  SET r.page = $page       // update with latest page seen
RETURN r
"""

################################################################################
# ----------------------------- constraint helpers ---------------------------
################################################################################

def _setup_constraints(tx):
    """Ensure only `key` has a uniqueness constraint for Entity nodes."""
    # Fetch existing constraints (Neo4j >=4.4 uses SHOW CONSTRAINTS)
    existing = tx.run("SHOW CONSTRAINTS").data()
    for c in existing:
        if 'Entity' in c.get('labelsOrTypes', []) and 'name' in c.get('properties', []) and c.get('type').startswith('UNIQUENESS'):
            # Attempt to drop the constraint enforcing uniqueness on `name`
            tx.run(f"DROP CONSTRAINT {c['name']} IF EXISTS")
    # Create / ensure the key constraint
    tx.run(CREATE_KEY_CONSTRAINT)

################################################################################
# ----------------------------- ingestion -------------------------------------
################################################################################

def ingest_file(session, fp: Path, batch_size: int):
    triples: List[Dict] = json.loads(fp.read_text())
    company = fp.name.split("_", 1)[0]  # Apple, Broadcom, …
    formatted = [
        {
            # Use predicate as context to ensure different triples get unique entities
            "s_key": normalize(t["subject"], company, t["predicate"]),
            "s_name": t["subject"],
            "o_key": normalize(t["object"], company, t["predicate"]),
            "o_name": t["object"],
            "pred": t["predicate"],
            "page": t.get("page", -1),
            "company": company,
        }
        for t in triples
    ]
    total = len(formatted)
    done = 0
    for batch in chunked(formatted, batch_size):
        session.execute_write(_upsert_batch, batch)
        done += len(batch)
        print(f"  {fp.name}: {done:,}/{total:,}", end="\r")
    print(f"  {fp.name}: {total:,} ✓")


def _upsert_batch(tx, batch: List[Dict]):
    for rec in batch:
        # Include company in node creation
        tx.run(UPSERT_NODE, key=rec["s_key"], name=rec["s_name"], company=rec["company"])
        tx.run(UPSERT_NODE, key=rec["o_key"], name=rec["o_name"], company=rec["company"])
        tx.run(
            UPSERT_REL,
            s_key=rec["s_key"],
            o_key=rec["o_key"],
            pred=rec["pred"],
            page=rec["page"],
            company=rec["company"],
        )

################################################################################
# ----------------------------- main ------------------------------------------
################################################################################

def main():
    pa = argparse.ArgumentParser(description="Ingest canonical triplets into Neo4j")
    pa.add_argument("--triplet-dir", type=Path, default=Path("canonical_output/canonical_triplets"))
    pa.add_argument("--batch-size", type=int, default=1000)
    args = pa.parse_args()

    if not args.triplet_dir.exists():
        raise SystemExit(f"Triplet dir not found: {args.triplet_dir}")

    url      = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise SystemExit("NEO4J_PASSWORD env var missing – add it to .env")

    print(f"→ Connecting to Neo4j @ {url} as {user}")
    driver = GraphDatabase.driver(url, auth=(user, password))

    with driver.session() as ses:
        ses.execute_write(lambda tx: _setup_constraints(tx))
        print("✓ constraints ensured")

        for fp in sorted(args.triplet_dir.glob("*_triplets_canonical.json")):
            ingest_file(ses, fp, args.batch_size)

    print("✅  Finished – open Neo4j Browser and explore your multi-company KG")

if __name__ == "__main__":
    main()
