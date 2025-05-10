# Financial Knowledge Graph Extraction Pipeline

This project implements a complete pipeline for extracting financial knowledge from corporate financial documents (10-K reports) and converting it into a structured knowledge graph for analysis and insights.

## Overview

The pipeline extracts financial knowledge triplets (subject-relationship-object) from financial documents and builds a knowledge graph representing the relationships between entities such as companies, products, financial metrics, geographic segments, and more.

## Pipeline Components

### 1. Data Processing

- **Input Data**: Financial documents in text format (primarily 10-K reports)
- **Parsing**: Text parsing to extract structured content
- **JSONL Format**: Documents stored as JSON Lines format for processing

### 2. Triplet Extraction

- **GPT-based Extraction**: Uses LLM-based extraction to identify financial triplets
- **Entity Recognition**: Identifies companies, products, services, financial metrics, etc.
- **Relationship Extraction**: Captures relationships between entities
- **Numerical Handling**: Extracts numerical values with proper units and scale
- **Temporal Context**: Preserves temporal information for financial data

### 3. Knowledge Graph Construction

- **Graph Building**: Converts triplets into a structured knowledge graph
- **Node Types**: Creates nodes for different entity types (company, product, financial_metric, etc.)
- **Edge Types**: Creates edges with relationship types and attributes
- **Metadata**: Stores confidence scores, numerical values, temporal information, etc.

### 4. Analysis and Visualization

- **Graph Statistics**: Calculates centrality, density, clustering, etc.
- **Entity Analysis**: Identifies most central entities and relationships
- **Financial Analysis**: Extracts financial insights like revenue distributions
- **Visualizations**: Creates graph visualizations for different views of the data
- **HTML Reports**: Generates comprehensive analysis reports

## Directory Structure

```
igenius/
├── data/
│   ├── parsed/               # Parsed financial documents
│   ├── triples/              # Extracted triplets in CSV format
│   ├── kg/                   # Knowledge graph files
│   ├── visualizations/       # Graph visualizations
│   └── reports/              # Analysis reports
├── scripts/
│   ├── batch_process.py      # Batch processing of documents
│   ├── gpt_extract.py        # LLM-based triplet extraction
│   ├── 03_build_kg.py        # Knowledge graph construction
│   ├── visualize_kg.py       # Graph visualization
│   └── generate_kg_report.py # Analysis report generation
```

## Extracted Information

The system extracts various types of financial information including:

1. **Company Products and Services**: All products and services offered by the company
2. **Financial Metrics**: Revenue, income, expenses, assets, liabilities, etc.
3. **Geographic Segments**: Regional operations and their financial performance
4. **Distribution Channels**: Direct vs. indirect distribution and their percentages
5. **Employee Information**: Employee counts and related data
6. **Temporal Data**: Associated fiscal years and quarters for financial metrics

## Knowledge Graph Format

The knowledge graph is saved in multiple formats:

- **GraphML**: Standard format for graph exchange (financial_kg.graphml)
- **GML**: Alternative graph format for compatibility (financial_kg.gml)
- **CSV**: Node and edge lists in CSV format for easy inspection (kg_nodes.csv, kg_edges.csv)
- **JSON**: Metadata and statistics (kg_metadata.json)

## Usage

### 1. Process Financial Documents

```bash
python scripts/batch_process.py data/parsed/COMPANY_FILE.jsonl
```

### 2. Build Knowledge Graph

```bash
python scripts/03_build_kg.py
```

### 3. Generate Visualizations

```bash
python scripts/visualize_kg.py
```

### 4. Generate Analysis Reports

```bash
python scripts/generate_kg_report.py
```

### 5. Launch Interactive Streamlit App

```bash
# Use the convenience script
chmod +x run_app.sh
./run_app.sh

# Or run directly
streamlit run scripts/kg_streamlit_app.py
```

## Streamlit Application

The Streamlit app provides an interactive interface to explore the knowledge graph:

- **Overview Page**: High-level statistics and distributions
- **Interactive Graph**: Dynamic network visualization with filtering options
- **Entity Explorer**: Search and explore specific entities and their relationships
- **Financial Metrics**: Analysis of financial data extracted from documents
- **Question Answering**: RAG-based Q&A using the knowledge graph

### Question Answering with RAG

The app includes a graph-based RAG (Retrieval Augmented Generation) system that:
1. Converts your question into an embedding
2. Finds the most relevant nodes and edges in the knowledge graph
3. Creates a context from the retrieved graph elements
4. Constructs an answer based on the relevant information
5. Visualizes the subgraph of relevant information

Example questions:
- "What products does Apple offer?"
- "What is Apple's revenue in 2024?"
- "How many employees does Apple have?"
- "What are Apple's distribution channels?"

## Sample Output

### Knowledge Graph Statistics

- **Nodes**: 178 (Entities like companies, products, services, etc.)
- **Edges**: 402 (Relationships between entities)
- **Entity Types**: 10 (company, product, service, financial_metric, etc.)
- **Relationship Types**: 17 (offers_product, reports_revenue, etc.)
- **Numerical Data**: 45.8% of edges contain numerical values
- **Temporal Data**: 45.8% of edges contain temporal information

### Node Type Distribution

- **Products**: 96 nodes (53.9%)
- **Financial Metrics**: 52 nodes (29.2%)
- **Services**: 13 nodes (7.3%)
- **Geographic Segments**: 5 nodes (2.8%)
- **Other Types**: 12 nodes (6.7%)

### Relationship Types

- **offers_product**: 63 edges (15.7%)
- **offers_service**: 47 edges (11.7%)
- **reports_income**: 15 edges (3.7%)
- **reports_assets**: 11 edges (2.7%)
- **Other relationships**: 266 edges (66.2%)

## Visualization Examples

The pipeline generates several visualizations:

1. **Full Graph**: Complete knowledge graph with all nodes and edges
2. **Important Nodes**: Top 20 most central nodes
3. **Apple Subgraph**: Nodes directly connected to Apple Inc.
4. **Financial Metric Subgraph**: Financial metrics and their connections

## Analysis Reports

The system generates comprehensive analysis reports:

- **JSON Report**: Complete data for programmatic analysis
- **HTML Report**: Human-readable report with tables and statistics

## Dependencies

- Python 3.8+
- NetworkX
- Pandas
- Matplotlib
- OpenAI API (for extraction)

## Future Improvements

- Support for more financial document types (quarterly reports, earnings calls)
- Enhanced entity resolution and deduplication
- Interactive web-based visualization
- Temporal analysis across multiple reporting periods
- Integration with financial databases for context 