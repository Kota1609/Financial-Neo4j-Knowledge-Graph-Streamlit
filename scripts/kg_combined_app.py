
import os
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import tempfile
import textwrap
from neo4j import GraphDatabase
from pyvis.network import Network
import json

try:
    from langchain_openai import OpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

st.set_page_config(
    page_title="Financial 10K Knowledge Graph Explorer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .answer-card {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
    }
    .header-text {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Neo4j connection settings
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to connect to Neo4j
@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        return None

# Function to search entities in the knowledge graph
def search_entities(driver, search_term, company=None):
    with driver.session() as session:
        # Search for entities containing the search term, optionally filtered by company
        if company and company != "All Companies":
            query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($search_term) AND e.company = $company
                RETURN e.key as key, e.name as name, e.company as company
                LIMIT 15
            """
            result = session.run(query, search_term=search_term, company=company)
        else:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($search_term)
                RETURN e.key as key, e.name as name, e.company as company
                LIMIT 15
            """
            result = session.run(query, search_term=search_term)
        
        return [(record["key"], record["name"], record.get("company", "Unknown")) for record in result]

# Function to get graph data for visualization
def get_entity_graph(driver, entity_key, depth=1, limit=50):
    with driver.session() as session:
        # Get subgraph centered on the given entity
        query = """
        MATCH path = (e:Entity {key: $entity_key})-[r:FACT*1..%d]-(related)
        RETURN path
        LIMIT %d
        """ % (depth, limit)
        
        result = session.run(query, entity_key=entity_key)
        
        # Extract nodes and relationships
        nodes = set()
        edges = []
        
        for record in result:
            path = record["path"]
            path_nodes = path.nodes
            path_relationships = path.relationships
            
            # Add nodes
            for node in path_nodes:
                node_id = node.element_id
                node_props = dict(node.items())
                company = node_props.get("company", "Unknown")
                nodes.add((node_id, node_props.get("key", ""), node_props.get("name", ""), company))
            
            # Add edges
            for rel in path_relationships:
                source_id = rel.start_node.element_id
                target_id = rel.end_node.element_id
                rel_props = dict(rel.items())
                predicate = rel_props.get("predicate", "")
                page = rel_props.get("page", "")
                company = rel_props.get("company", "Unknown")
                edges.append((source_id, target_id, predicate, page, company))
        
        return list(nodes), edges

# Display the visualization tab
def show_visualization_tab(driver):
    st.subheader("Knowledge Graph Explorer")
    
    # Simple company filter and search
    col1, col2 = st.columns([1, 2])
    
    with col1:
        companies = ["All Companies", "Apple", "Google", "Microsoft", "Intel", "Nvidia", "Qualcomm", "Micron", "Broadcom"]
        selected_company = st.selectbox("Company:", companies, key="viz_company")
    
    with col2:
        search_term = st.text_input("Search:", placeholder="Type to search entities (product, revenue, CEO, etc.)")
    
    # Handle search
    if search_term:
        with st.spinner("Searching..."):
            search_results = search_entities(driver, search_term, selected_company)
        
        if search_results:
            st.success(f"Found {len(search_results)} matching entities")
            
            # Create a clean table display
            result_df = pd.DataFrame(search_results, columns=["Key", "Name", "Company"])
            st.dataframe(result_df.drop(columns=["Key"]), use_container_width=True)
            
            # Let user select entity
            selected_index = st.selectbox(
                "Select entity to visualize:",
                options=range(len(search_results)),
                format_func=lambda i: f"{search_results[i][1]} ({search_results[i][2]})"
            )
            
            if st.button("Visualize Graph", type="primary"):
                with st.spinner("Generating graph..."):
                    # Get the selected entity key and name
                    entity_key = search_results[selected_index][0]
                    entity_name = search_results[selected_index][1]
                    
                    # Get graph data with fixed depth and limit
                    nodes, edges = get_entity_graph(driver, entity_key, 2, 30)
                    
                    if nodes and edges:
                        st.subheader(f"Knowledge Graph: {entity_name}")
                        
                        # Generate visualization
                        graph_html = simple_graph_visualization(nodes, edges)
                        components.html(graph_html, height=700)
    
                        # Display facts
                        with st.expander("Facts"):
                            fact_data = []
                            for source_id, target_id, predicate, page, company in edges:
                                source_name = next((name for node_id, _, name, _ in nodes if node_id == source_id), "")
                                target_name = next((name for node_id, _, name, _ in nodes if node_id == target_id), "")
                                fact_data.append({
                                    "Subject": source_name,
                                    "Predicate": predicate,
                                    "Object": target_name,
                                    "Company": company
                                })
                            
                            if fact_data:
                                st.dataframe(pd.DataFrame(fact_data), use_container_width=True)
                    else:
                        st.warning("No relationships found for this entity.")
        else:
            st.info("No entities found matching your search term.")

def simple_graph_visualization(nodes, edges):
    """Create a simple graph visualization using pyvis"""
    net = Network(height="700px", width="100%", directed=True, notebook=True)
    
    # Color map for companies
    company_colors = {
        "Apple": "#A2AAAD",
        "Microsoft": "#00A4EF",
        "Google": "#4285F4",
        "Intel": "#0071C5",
        "Nvidia": "#76B900",
        "Qualcomm": "#3253DC",
        "Broadcom": "#CC092F",
        "Micron": "#224099",
        "Unknown": "#607D8B"
    }
    
    # Add nodes
    for node_id, _, node_name, company in nodes:
        color = company_colors.get(company, company_colors["Unknown"])
        net.add_node(node_id, label=node_name[:20] + "..." if len(node_name) > 20 else node_name, 
                    title=f"{node_name} ({company})", color=color)
    
    # Add edges
    for source_id, target_id, predicate, page, company in edges:
        net.add_edge(source_id, target_id, title=predicate)
    
    # Generate and return HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as f2:
            return f2.read()

# Function to retrieve relevant facts from knowledge graph based on entities in question
def retrieve_facts_from_kg(driver, question, limit=20, company=None):
    facts = []  # gather facts incrementally

    #############################
    # Heuristic 1 – Highest revenue
    #############################
    if re.search(r"highest|largest|biggest", question, re.IGNORECASE) and re.search(r"revenue|sales", question, re.IGNORECASE):
        with driver.session() as session:
            # Get the list of all companies
            company_query = """
                MATCH ()-[r:FACT]->()
                WHERE r.company IS NOT NULL
                RETURN DISTINCT r.company as company
            """
            companies = [record["company"] for record in session.run(company_query)]
            
            # Build a comprehensive list of revenue facts for all companies
            revenue_rows = []
            
            # For each company, find their top revenue numbers
            for company_name in companies:
                if not company_name:
                    continue
                    
                # Try to fetch all revenue facts for this company
                query = """
                    MATCH (c:Entity)-[r:FACT]->(o:Entity)
                    WHERE r.company = $company_name AND 
                          (toLower(r.predicate) CONTAINS 'revenue' OR 
                           toLower(r.predicate) CONTAINS 'sales' OR
                           toLower(r.predicate) = 'has total revenue' OR
                           toLower(r.predicate) CONTAINS 'net sales')
                    RETURN c.name as entity, r.predicate as predicate, o.name as value, 
                           r.page as page, r.company as company
                    LIMIT 10
                """
                records = session.run(query, company_name=company_name)
                
                # Process revenue values for this company
                for rec in records:
                    val = rec["value"]
                    company_name = rec["company"]  # Get company directly from relationship
                    
                    # Clean and standardize the value for display
                    clean_val = val.replace("$", "").strip()
                    
                    # Extract numeric part with regex
                    m = re.search(r"([0-9,.]+)", clean_val)
                    if m and m.group(1):
                        try:
                            # Handle numbers with commas and potential suffixes
                            num_str = m.group(1).replace(",", "")
                            num = float(num_str)
                            
                            # Check for million/billion/trillion
                            if "million" in val.lower():
                                num *= 1e6
                                clean_suffix = "million"
                            elif "billion" in val.lower():
                                num *= 1e9
                                clean_suffix = "billion"
                            elif "trillion" in val.lower():
                                num *= 1e12
                                clean_suffix = "trillion"
                            else:
                                clean_suffix = ""
                                
                            # Create a clean display value
                            display_val = f"${num_str} {clean_suffix}".strip()
                                
                            revenue_rows.append((
                                rec["entity"],                # Subject entity name
                                rec["predicate"],            # Predicate 
                                display_val,                 # Cleaned display value
                                num,                         # Numeric value for sorting
                                company_name,                # Company source
                                rec["page"]                  # Page reference
                            ))
                            
                        except ValueError:
                            # If we can't parse it as a number, still include it
                            revenue_rows.append((
                                rec["entity"],
                                rec["predicate"],
                                val,
                                0,  # Default numeric value
                                company_name,
                                rec["page"]
                            ))
                
            # Now get company descriptions too
            # For each company, get some context
            for company_name in companies:
                if not company_name:
                    continue
                    
                # Get basic info about the company
                context_query = """
                    MATCH (c:Entity)-[r:FACT]->(o:Entity)
                    WHERE r.company = $company_name AND 
                          (toLower(r.predicate) CONTAINS 'business' OR 
                           toLower(r.predicate) CONTAINS 'description' OR
                           toLower(r.predicate) CONTAINS 'operations' OR
                           toLower(r.predicate) CONTAINS 'segment')
                    RETURN c.name as subject, r.predicate as predicate, o.name as object, 
                           r.page as page, r.company as company
                    LIMIT 2
                """
                context_records = session.run(context_query, company_name=company_name)
                
                for rec in context_records:
                    company_rec = rec.get('company', company_name)
                    facts.append({
                        'subject': rec['subject'],
                        'predicate': rec['predicate'],
                        'object': rec['object'],
                        'company': company_rec if company_rec else company_name,
                        'page': rec['page'] if rec['page'] is not None else 'N/A'
                    })
                    
            if revenue_rows:
                # Sort by numeric value desc and keep top N
                revenue_rows.sort(key=lambda x: x[3], reverse=True)
                
                # Add all revenue facts
                for row in revenue_rows:
                    facts.append({
                        'subject': row[0],
                        'predicate': row[1],
                        'object': row[2],
                        'company': row[4] if row[4] else row[0].split()[0],  # Try to derive company from subject
                        'page': row[5] if row[5] else 'N/A'
                    })

    #############################
    # Heuristic 2 – Segment composition (e.g., Europe segment countries)
    #############################
    seg_match = re.search(r"(\w+)'s?\s+([A-Za-z ]+) segment", question, re.IGNORECASE)
    if seg_match:
        comp = seg_match.group(1)
        segment = seg_match.group(2).strip()
        with driver.session() as session:
            query = """
                MATCH (s:Entity)-[r:FACT]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($segment) AND toLower(r.predicate) CONTAINS 'includes'
                RETURN s.name as subject, r.predicate as predicate, o.name as object, r.company as company, r.page as page
            """
            for rec in session.run(query, segment=segment):
                # Optionally filter by company if specified
                if company and company != 'All Companies' and rec['company'] != company:
                    continue
                facts.append({
                    'subject': rec['subject'],
                    'predicate': rec['predicate'],
                    'object': rec['object'],
                    'company': rec['company'] if rec['company'] else 'Unknown',
                    'page': rec['page'] if rec['page'] is not None else 'N/A'
                })

    # Extract potential entity names from the question
    words = re.findall(r'\b[A-Z][A-Za-z]*\b', question)
    company_names = ['Apple', 'Broadcom', 'Google', 'Intel', 'Micron', 'Microsoft', 'Nvidia', 'Qualcomm']
    
    # Always include company names in the search
    for company_name in company_names:
        if company_name.lower() in question.lower() and company_name not in words:
            words.append(company_name)
    
    # If a specific company is selected, focus the search on that company
    if company and company != "All Companies" and company not in words:
        words.append(company)
    
    with driver.session() as session:
        # Build the WHERE clause for company filtering
        company_filter = f"AND r.company = '{company}'" if company and company != "All Companies" else ""
        
        # First try to find facts related to specific entities in the question
        for word in words:
            query = f"""
                MATCH (s:Entity)
                WHERE toLower(s.name) CONTAINS toLower($term)
                MATCH (s)-[r:FACT]->(o)
                WHERE 1=1 {company_filter}
                RETURN s.name as subject, r.predicate as predicate, o.name as object, 
                       r.company as company, r.page as page
                LIMIT $limit
            """
            
            result = session.run(query, term=word, limit=limit)
            
            for record in result:
                company_value = record.get('company')
                facts.append({
                    'subject': record['subject'],
                    'predicate': record['predicate'],
                    'object': record['object'],
                    'company': company_value if company_value else record['subject'].split()[0],
                    'page': record['page'] if record['page'] is not None else 'N/A'
                })
            
            # Also get facts where the entity is the object
            query = f"""
                MATCH (s:Entity)-[r:FACT]->(o:Entity)
                WHERE toLower(o.name) CONTAINS toLower($term) {company_filter}
                RETURN s.name as subject, r.predicate as predicate, o.name as object, 
                       r.company as company, r.page as page
                LIMIT $limit
            """
            
            result = session.run(query, term=word, limit=limit)
            
            for record in result:
                company_value = record.get('company')
                facts.append({
                    'subject': record['subject'],
                    'predicate': record['predicate'],
                    'object': record['object'],
                    'company': company_value if company_value else record['subject'].split()[0],
                    'page': record['page'] if record['page'] is not None else 'N/A'
                })
        
        # If we didn't find facts based on specific entities, try a more general search
        if not facts or len(facts) < 5:  # If we have very few facts, augment with general search
            for word in question.lower().split():
                if len(word) > 3:  # Skip short words
                    query = f"""
                        MATCH (s)-[r:FACT]->(o)
                        WHERE 
                            (toLower(s.name) CONTAINS toLower($term) OR
                            toLower(r.predicate) CONTAINS toLower($term) OR
                            toLower(o.name) CONTAINS toLower($term))
                            {company_filter}
                        RETURN s.name as subject, r.predicate as predicate, o.name as object, 
                               r.company as company, r.page as page
                        LIMIT $limit
                    """
                    
                    result = session.run(query, term=word, limit=limit)
                    
                    for record in result:
                        company_value = record.get('company')
                        facts.append({
                            'subject': record['subject'],
                            'predicate': record['predicate'],
                            'object': record['object'],
                            'company': company_value if company_value else record['subject'].split()[0],
                            'page': record['page'] if record['page'] is not None else 'N/A'
                        })
                    
                    # Continue searching even if we found some facts to gather more context
    
    # Remove duplicates while preserving order
    unique_facts = []
    seen = set()
    for fact in facts:
        fact_tuple = (fact['subject'], fact['predicate'], fact['object'])
        if fact_tuple not in seen:
            seen.add(fact_tuple)
            unique_facts.append(fact)
    
    return unique_facts

# Function to format facts as text for the LLM
def format_facts_as_context(facts):
    if not facts:
        return "No relevant facts found in the knowledge base."
    
    formatted_facts = []
    for i, fact in enumerate(facts, 1):
        # Format each fact as a sentence
        formatted_fact = f"{i}. {fact['subject']} {fact['predicate']} {fact['object']} (Company: {fact['company']}, Page: {fact['page']})."
        formatted_facts.append(formatted_fact)
    
    return "\n".join(formatted_facts)

# Function to generate an answer using LLM with retrieved facts
def generate_answer(question, facts, api_key):
    # Format the facts as context
    context = format_facts_as_context(facts)
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=textwrap.dedent("""
        You are a financial analyst who answers questions about various technology companies based on facts from 
        their 10-K reports stored in a knowledge graph. Use only the facts provided to answer the question.
        
        Facts from the 10-K Reports Knowledge Graph:
        {context}
        
        Question: {question}
        
        Provide a comprehensive answer based only on the facts above. If the facts don't contain enough information 
        to answer the question completely, state what's missing. Don't invent information that isn't in the facts.
        
        IMPORTANT: When mentioning numeric values like revenue, always format them clearly and consistently,
        using standard notation. For example, write "$274.5 billion" rather than spreading out the digits.
        Always convert units to be consistent when comparing values.
        
        Keep your answer concise and limit it to 150 words maximum.
        
        Answer:
        """)
    )
    
    # Create the LLM
    try:
        from langchain_openai import ChatOpenAI
        
        # Use ChatOpenAI instead of OpenAI for GPT-4o
        llm = ChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            model_name="gpt-4o"
        )
        
        # Create and run the chain
        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt_template
            | llm
        | StrOutputParser()
        )
        
        response = chain.invoke({"question": question, "context": context})
        return response
            
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Display the QA tab
def show_qa_tab(driver, api_key):
    st.subheader("Question Answering")
    
    # Company selection
    companies = ["All Companies", "Apple", "Google", "Microsoft", "Intel", "Nvidia", "Qualcomm", "Micron", "Broadcom"]
    selected_company = st.selectbox("Focus on company:", companies, key="qa_company")
            
    # Example questions based on selected company
    st.markdown("### Example questions")
    
    if selected_company == "All Companies":
        example_questions = [
            "Which company has the highest revenue?",
            "What are the main risks tech companies face?",
            "Compare Apple and Microsoft's businesses",
            "What products does Apple offer?",
            "Who is the CEO of Nvidia?"
        ]
    else:
        example_questions = [
            f"What is {selected_company}'s business?",
            f"How much revenue did {selected_company} report?",
            f"What are {selected_company}'s main products?",
            f"What risks does {selected_company} face?",
            f"Who is the CEO of {selected_company}?"
        ]
    
    cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        if cols[i].button(q, key=f"btn_{i}"):
            st.session_state.question = q
    
    # Question input
    if "question" not in st.session_state:
        st.session_state.question = ""
    
    question = st.text_input("Enter your question:", key="question")
    
    if st.button("Ask Question", type="primary", disabled=not (question and api_key)) and question:
        with st.spinner("Retrieving information from knowledge graph..."):
            # Retrieve facts
            facts = retrieve_facts_from_kg(driver, question, company=selected_company)
            
            # Display facts
            st.subheader("Retrieved Facts")
            
            if facts:
                facts_df = pd.DataFrame(facts)
                st.dataframe(facts_df, use_container_width=True)
            else:
                st.info("No relevant facts found for this question.")
        
        # Generate answer
        with st.spinner("Generating answer..."):
            st.subheader("Answer")
            answer = generate_answer(question, facts, api_key)
            
            # Apply some formatting to clean up the response
            answer = answer.replace("$", "\\$")  # Escape dollar signs for Markdown
            
            # Display answer in a styled card with proper markdown rendering
            st.markdown(f"""
            <div class="answer-card">
                {answer}
            </div>
            """, unsafe_allow_html=True)

# Main App
def main():
    st.title("🧠 Financial 10K Knowledge Graph Explorer")
    st.markdown("Explore and query the financial knowledge graph built from 10-K reports")
    
    # Connect to Neo4j
    driver = get_neo4j_driver()
    
    if not driver:
        st.error("Failed to connect to Neo4j. Please check your connection settings.")
        return
        
    # Check for API key (needed for QA)
    api_key = st.sidebar.text_input("OpenAI API Key", value=OPENAI_API_KEY if OPENAI_API_KEY else "", type="password")
        
    # Display graph stats in sidebar
    with st.sidebar.expander("Knowledge Graph Info", expanded=True):
        with driver.session() as session:
            # Count entities
            entity_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            
            # Count facts
            fact_count = session.run("MATCH ()-[r:FACT]->() RETURN count(r) as count").single()["count"]
            
            # Count companies
            company_count = session.run("MATCH ()-[r:FACT]->() RETURN count(DISTINCT r.company) as count").single()["count"]
            
            # Display statistics
            st.write(f"📊 **Entities:** {entity_count:,}")
            st.write(f"🔗 **Facts:** {fact_count:,}")
            st.write(f"🏢 **Companies:** {company_count}")
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["🔍 Knowledge Graph Visualization", "❓ Question Answering"])
    
    with tab1:
        show_visualization_tab(driver)
    
    with tab2:
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to use the QA functionality.")
        else:
            show_qa_tab(driver, api_key)

if __name__ == "__main__":
    main() 