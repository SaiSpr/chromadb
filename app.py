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
import re

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
        return result  # This should be JSON with extracted biomarkers and filters
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# ✅ Initialize ChromaDB
# -------------------------------
CHROMA_DB_DIR = "./"  # Ensure correct folder path

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("clinical_trials")

# Load embedding model (Force CPU for Hugging Face Spaces)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# -------------------------------
# ✅ Helper Functions for Filtering
# -------------------------------

def parse_filter_criteria(filter_value):
    """
    Parses filter criteria (>, >=, <, <=, !=, =) into ChromaDB-supported format.
    Example: 
      ">=50" → ("$gte", 50)
      "<60" → ("$lt", 60)
    """
    import re

    match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
    if match:
        operator_map = {
            ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"
        }
        op, value = match.groups()
        return operator_map.get(op), int(value)
    return None, None  # Return None if no valid filter is found


def build_metadata_filter(parsed_input):
    """
    Constructs a ChromaDB-compatible metadata filter using `$and` for multiple conditions.
    """
    filters = []

    # Country Filter (Exact Match)
    if parsed_input.get("country"):
        filters.append({"country": {"$eq": parsed_input["country"]}})

    # Study Size Filter (Handles >, >=, <, <=, !=, =)
    if parsed_input.get("study_size"):
        operator, value = parse_filter_criteria(parsed_input["study_size"])
        if operator:
            filters.append({"count": {operator: value}})

    # Age Filter (Handles >, >=, <, <=, !=, =)
    if parsed_input.get("ages"):
        operator, value = parse_filter_criteria(parsed_input["ages"])
        if operator:
            filters.append({"age": {operator: value}})

    # Gender Filter (Matches "All" or the specified gender)
    if parsed_input.get("gender"):
        filters.append({"sex": {"$in": ["ALL", parsed_input["gender"].upper()]}})

    # Status Filter (Exact Match)
    if parsed_input.get("status"):
        filters.append({"overallStatus": {"$eq": parsed_input["status"].upper()}})

    # Combining Filters
    if len(filters) == 1:
        return filters[0]  # Single filter, no need for `$and`
    elif len(filters) > 1:
        return {"$and": filters}  # Apply multiple conditions
    else:
        return None  # No filters applied



# -------------------------------
# ✅ Query ChromaDB Based on Extracted JSON
# -------------------------------

def flatten_list(nested_list):
    """Flattens a list of lists into a single list."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def query_chromadb(parsed_input):
    """Search ChromaDB using extracted biomarker JSON & strict metadata filters."""

    metadata_filters = build_metadata_filter(parsed_input)

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

    # Query ChromaDB with metadata filtering
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,  # Fetch top 10 matches
        where=metadata_filters  # Apply strict filters
    )

    # Convert results into a DataFrame
    if results and "metadatas" in results and results["metadatas"]:
        df = pd.DataFrame(results["metadatas"][0])
        return df
    else:
        return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# -------------------------------
# ✅ Convert Data to Static Table Format
# -------------------------------
def format_results_as_table(df, extracted_biomarkers):
    """Format clinical trial results into a structured DataFrame for display."""
    
    table_data = []
    
    for _, row in df.iterrows():
        table_data.append([
            f"[{row['nctId']}](https://clinicaltrials.gov/study/{row['nctId']})",  # Hyperlinked ID
            ", ".join(flatten_list(extracted_biomarkers.get("inclusion_biomarker", []))),  # Biomarker Match
            row["condition"],
            row["overallStatus"],
            row["count"],
            row["sex"],
            row["startDate"],
            row["country"]
        ])

    table_df = pd.DataFrame(
        table_data,
        columns=["Trial ID", "Biomarker", "Condition", "Status", "Study Size", "Gender", "Start Date", "Country"]
    )
    
    return table_df

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
user_input = st.text_area("Enter clinical trial eligibility criteria:", placeholder="e.g., BRAF mutation, age > 50, gender=male, country=China")

if st.button("🔍 Extract Biomarkers & Find Trials"):
    if user_input.strip():
        # Extract Biomarkers
        st.markdown("### 🧬 Extracted Biomarkers & Filters:")
        response = get_model_response(user_input)

        if isinstance(response, dict):
            st.json(response)  # Show extracted biomarkers & filters
            
            # Query ChromaDB with extracted biomarkers
            st.markdown("### 🔍 Matching Clinical Trials:")
            trial_results = query_chromadb(response)
            
            if not trial_results.empty:
                formatted_results = format_results_as_table(trial_results, response)
                st.table(formatted_results)  # Display as static table
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

