__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3 as sqlite3

import streamlit as st
import pandas as pd
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from gradio_client import Client

# -------------------------------
# ✅ Initialize Hugging Face Client
# -------------------------------
HF_CLIENT = Client("SaiPrakashTut/Galileo_twostep_gpu")

def get_model_response(input_text):
    """Send input text to Hugging Face model and return the response."""
    try:
        result = HF_CLIENT.predict(
            input_text=input_text,
            api_name="/extract_criteria"
        )
        return result  # Should be JSON with extracted biomarkers and filters
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# ✅ Initialize ChromaDB
# -------------------------------
CHROMA_DB_DIR = "./"  # Ensure correct folder path
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("clinical_trials")

# Load embedding model (Force CPU for Hugging Face Spaces)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -------------------------------
# ✅ Query ChromaDB with Filters
# -------------------------------
def flatten_list(nested_list):
    """Flattens a nested list into a single list."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def apply_filters(df, filters):
    """Apply filters dynamically based on extracted JSON."""
    
    if filters.get("country"):
        df = df[df["country"].str.contains(filters["country"], case=False, na=False)]

    if filters.get("study_size"):
        try:
            study_size = int(filters["study_size"])
            df = df[df["count"] >= study_size]
        except ValueError:
            pass  # Ignore invalid values

    if filters.get("gender"):
        gender_filter = filters["gender"].lower()
        df = df[df["sex"].str.lower().isin(["all", gender_filter])]

    return df

def query_chromadb(parsed_input):
    """Search ChromaDB based on extracted biomarker JSON criteria."""
    query_text = f"""
    Biomarkers: {', '.join(flatten_list(parsed_input.get('inclusion_biomarker', [])))}
    Exclusions: {', '.join(flatten_list(parsed_input.get('exclusion_biomarker', [])))}
    Status: {parsed_input.get('status', '')}
    Study Size: {parsed_input.get('study_size', '')}
    Ages: {parsed_input.get('ages', '')}
    Gender: {parsed_input.get('gender', '')}
    Country: {parsed_input.get('country', '')}
    """

    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)

    # Query ChromaDB for matching trials
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5  # Fetch top 5 matches
    )

    # Convert results into a DataFrame
    if results and "metadatas" in results and results["metadatas"]:
        df = pd.DataFrame(results["metadatas"][0])
        df = apply_filters(df, parsed_input)  # Apply extracted filters
        return df
    else:
        return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# -------------------------------
# ✅ Generate HTML Table with Clickable Links
# -------------------------------
def generate_html_table(trial_results, biomarkers):
    """
    Generate an HTML table from trial results with:
    - Clickable Trial ID links
    - Highlighted biomarker in eligibility criteria
    """
    html_content = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        td a {
            color: #007bff;
            text-decoration: none;
        }
        td a:hover {
            text-decoration: underline;
        }
        .highlight {
            font-weight: bold;
            color: red;
        }
    </style>
    <table>
        <tr>
            <th>Trials ID</th>
            <th>Biomarker</th>
            <th>Condition</th>
            <th>Status</th>
            <th>Study Size</th>
            <th>Gender</th>
            <th>Start Date</th>
            <th>Country</th>
        </tr>
    """

    for _, row in trial_results.iterrows():
        trial_id = row["nctId"]
        trial_link = f"https://clinicaltrials.gov/study/{trial_id}"
        condition = row["condition"]
        status = row["overallStatus"]
        study_size = row["count"]
        gender = row["sex"]
        start_date = row["startDate"]
        country = row["country"]
        
        # Highlight biomarkers in eligibility text
        eligibility = row["eligibility"]
        for biomarker in biomarkers:
            eligibility = eligibility.replace(biomarker, f'<span class="highlight">{biomarker}</span>')

        html_content += f"""
        <tr>
            <td><a href="{trial_link}" target="_blank">{trial_id}</a></td>
            <td>{', '.join(biomarkers)}</td>
            <td>{condition}</td>
            <td>{status}</td>
            <td>{study_size}</td>
            <td>{gender}</td>
            <td>{start_date}</td>
            <td>{country}</td>
        </tr>
        """

    html_content += "</table>"
    return html_content

# -------------------------------
# ✅ Streamlit UI
# -------------------------------
st.set_page_config(page_title="🧬 Biomarker-Based Clinical Trial Finder", page_icon="🧬", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 Biomarker-Based Clinical Trial Finder 🏥</h1>
    <p style='text-align: center; font-size: 18px;'>Enter clinical text, extract biomarkers, and find matching trials!</p>
    <hr>
    """, unsafe_allow_html=True)

# User Input
user_input = st.text_area("Enter clinical trial eligibility criteria:", placeholder="e.g., BRAF mutation, age > 50, gender=male, country=United States")

if st.button("🔍 Extract Biomarkers & Find Trials"):
    if user_input.strip():
        # Extract Biomarkers
        st.markdown("### 🧬 Extracted Biomarkers & Filters:")
        response = get_model_response(user_input)

        if isinstance(response, dict):
            st.json(response)  # Show extracted biomarkers & filters
            
            # Extract biomarker list
            biomarkers = flatten_list(response.get('inclusion_biomarker', []))

            # Query ChromaDB with extracted biomarkers
            st.markdown("### 🔍 Matching Clinical Trials:")
            trial_results = query_chromadb(response)
            
            if not trial_results.empty:
                # Generate and display formatted HTML table
                st.markdown(generate_html_table(trial_results, biomarkers), unsafe_allow_html=True)
            else:
                st.warning("⚠️ No matching trials found!")
        else:
            st.error("❌ Error in fetching response. Please try again.")
    else:
        st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")










# import streamlit as st
# from gradio_client import Client

# # Initialize the Hugging Face client
# client = Client("SaiPrakashTut/Galileo_twostep_gpu")

# def get_model_response(input_text):
#     """Send input text to the Hugging Face model and return the response."""
#     try:
#         result = client.predict(
#             input_text=input_text,
#             api_name="/extract_criteria"
#         )
#         return result
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Streamlit UI
# st.set_page_config(page_title="🧬 Biomarker Extraction Tool 🏥", page_icon="🧬", layout="wide")

# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 Biomarker Extraction Tool 🏥</h1>
#     <p style='text-align: center; font-size: 18px;'>Extract genomic biomarkers from clinical trial texts!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# # User input
# user_input = st.text_area("Enter the clinical trial eligibility criteria below:", placeholder="e.g., braf or kras")

# if st.button("🔍 Extract Biomarkers"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers:")
#         response = get_model_response(user_input)
#         st.json(response)
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

