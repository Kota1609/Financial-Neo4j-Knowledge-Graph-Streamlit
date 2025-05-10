# Knowledge Graph Explorer App

This Streamlit application provides a comprehensive interface for exploring, visualizing, and querying the Apple 10-K Knowledge Graph stored in Neo4j.

## Features

The application has three main tabs:

1. **Explorer** - Get an overview of the knowledge graph with statistics, charts, and samples
2. **Visualization** - Interactive graph visualization allowing you to explore entity relationships
3. **Question Answering** - Ask natural language questions and get answers based on the knowledge graph

## Prerequisites

- Neo4j database running with the knowledge graph loaded using `scripts/build_kg.py`
- Python 3.8+ environment with required packages installed
- OpenAI API key for the QA functionality

## Setup

1. Ensure that your Neo4j database is running with the knowledge graph already loaded:

```bash
# Check that your knowledge graph has been successfully imported
python3 scripts/check_kg.py
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Set up environment variables for Neo4j and OpenAI:

```bash
export NEO4J_URL="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export OPENAI_API_KEY="your-openai-api-key"  # Only needed for the QA functionality
```

## Running the App

Start the Streamlit app with:

```bash
streamlit run kg_combined_app.py
```

## Using the App

### Explorer Tab
- View overall statistics about your knowledge graph
- See charts of the most common predicates
- Browse random samples of entities and facts

### Visualization Tab
- Search for entities in the knowledge graph
- Visualize entity connections in an interactive graph
- Adjust visualization depth and limit the number of nodes shown
- View facts in a tabular format

### Question Answering Tab
- Ask questions about Apple using natural language
- See relevant facts retrieved from the knowledge graph
- Get AI-generated answers based on the retrieved facts

## Troubleshooting

- If you encounter Neo4j connection errors, make sure your database is running and credentials are correct
- For visualization issues, check that Pyvis is properly installed
- If the QA tab doesn't work, ensure your OpenAI API key is valid and has sufficient quota 