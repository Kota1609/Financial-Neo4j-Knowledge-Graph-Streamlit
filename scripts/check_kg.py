

import os
from neo4j import GraphDatabase

# Use the environment variables
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to Neo4j at: {NEO4J_URL}")
driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as session:
    # Count entities
    result = session.run("MATCH (n:Entity) RETURN count(n) as entityCount")
    entity_count = result.single()["entityCount"]
    print(f"Entity count: {entity_count}")
    
    # Count facts
    result = session.run("MATCH ()-[r:FACT]->() RETURN count(r) as factCount")
    fact_count = result.single()["factCount"]
    print(f"Fact count (all companies): {fact_count}")
    
    # Count facts per company
    print("\nFacts per company:")
    result = session.run("MATCH ()-[r:FACT]->() RETURN r.company as company, count(r) as c ORDER BY c DESC")
    for record in result:
        print(f"  {record['company']}: {record['c']}")
    
    # Sample entities
    print("\nSample entities:")
    result = session.run("MATCH (n:Entity) RETURN n.key, n.name LIMIT 5")
    for record in result:
        print(f"Key: {record['n.key']} - Name: {record['n.name']}")
    
    # Sample facts
    print("\nSample facts:")
    result = session.run(
        "MATCH (s)-[r:FACT]->(o) RETURN r.company as company, s.name as s, r.predicate as p, o.name as o, r.page as page LIMIT 5"
    )
    for rec in result:
        print(f"[{rec['company']}] {rec['s']} --{rec['p']}--> {rec['o']}  (page {rec['page']})")

driver.close() 