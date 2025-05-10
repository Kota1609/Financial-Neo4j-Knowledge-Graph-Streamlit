
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

def load_env():
    """Search upwards for a .env and load it."""
    here = Path(__file__).resolve().parent
    for p in [here / ".env", here.parent / ".env", Path(".env")]:
        if p.exists():
            load_dotenv(p)
            break

load_env()

def main():
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise SystemExit("NEO4J_PASSWORD env var missing – add it to .env")
    
    print(f"→ Connecting to Neo4j @ {url} as {user}")
    driver = GraphDatabase.driver(url, auth=(user, password))
    
    with driver.session() as session:
        # Delete all relationships first, then nodes
        print("Deleting all relationships...")
        session.run("MATCH ()-[r]-() DELETE r")
        
        print("Deleting all nodes...")
        session.run("MATCH (n) DELETE n")
        
        # Verify the database is empty
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"Database now contains {count} nodes.")
    
    print("✅ Database cleared successfully")

if __name__ == "__main__":
    main() 