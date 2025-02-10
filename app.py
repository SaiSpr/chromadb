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
# ✅ Normalization Functions
# -------------------------------
COUNTRY_MAPPING = {
    "us": "United States", "usa": "United States", "u.s.": "United States",
    "america": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "gb": "United Kingdom", "england": "United Kingdom",
    "de": "Germany", "fr": "France", "cn": "China", "chinese": "China",
    "india": "India", "in": "India", "canada": "Canada", "ca": "Canada",
    "aus": "Australia", "australia": "Australia"
}

GENDER_MAPPING = {
    "f": "Female", "female": "Female", "woman": "Female", "women": "Female",
    "m": "Male", "male": "Male", "man": "Male", "men": "Male",
    "all": "All", "both": "All", "any": "All", "unisex": "All"
}

STATUS_MAPPING = {
    "recruiting": "RECRUITING", "not recruiting": "ACTIVE_NOT_RECRUITING",
    "completed": "COMPLETED", "terminated": "TERMINATED",
    "withdrawn": "WITHDRAWN", "not yet recruiting": "NOT_YET_RECRUITING",
    "suspended": "SUSPENDED", "unknown": "UNKNOWN"
}

def normalize_text(text, mapping):
    """Normalize text input by mapping common variations."""
    text = text.strip().lower()
    return mapping.get(text, text.upper())  # Convert to uppercase if not found

def normalize_country(country):
    """Normalize country input using the COUNTRY_MAPPING dictionary."""
    return normalize_text(country, COUNTRY_MAPPING)

def normalize_gender(gender):
    """Normalize gender input using the GENDER_MAPPING dictionary."""
    return normalize_text(gender, GENDER_MAPPING)

def normalize_status(status):
    """Normalize status input using the STATUS_MAPPING dictionary."""
    return normalize_text(status, STATUS_MAPPING)

# -------------------------------
# ✅ Numeric Filter Parsing
# -------------------------------
def parse_filter_criteria(value):
    """Extracts operator and numeric value (e.g., '>50' → ('>', 50))"""
    match = re.match(r"(>=|<=|>|<|=)?\s*(\d+)", str(value).strip())
    if match:
        operator = match.group(1) if match.group(1) else "="  # Default to "="
        numeric_value = int(match.group(2))
        return operator, numeric_value
    return None, None  # No valid filter found

# -------------------------------
# ✅ Apply Filters to DataFrame
# -------------------------------
def apply_filters(df, parsed_input):
    """Applies metadata filtering dynamically on retrieved results from ChromaDB."""
    
    # ✅ Country Filter
    if parsed_input.get("country"):
        country_filter = normalize_country(parsed_input["country"])
        df = df[df["country"].str.lower() == country_filter.lower()]

    # ✅ Study Size Filter
    if parsed_input.get("study_size"):
        operator, value = parse_filter_criteria(parsed_input["study_size"])
        if operator and value is not None:
            df = df[eval(f"df['count'] {operator} {value}")]

    # ✅ Age Filter
    if parsed_input.get("ages"):
        operator, value = parse_filter_criteria(parsed_input["ages"])
        if operator and value is not None:
            df = df[eval(f"df['age'] {operator} {value}")]

    # ✅ Gender Filter (Matches "All" or Specific Gender)
    if parsed_input.get("gender"):
        gender_filter = normalize_gender(parsed_input["gender"])
        df = df[df["sex"].isin([gender_filter, "All"])]  # Allow "All" to match all genders

    # ✅ Status Filter
    if parsed_input.get("status"):
        status_filter = normalize_status(parsed_input["status"])
        df = df[df["overallStatus"].str.lower() == status_filter.lower()]

    return df

# -------------------------------
# ✅ Query ChromaDB Based on Extracted JSON
# -------------------------------
def flatten_list(nested_list):
    """Flattens a list of lists into a single list of strings."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def query_chromadb(parsed_input):
    """Search ChromaDB based on extracted biomarker JSON criteria."""

    # Ensure biomarkers are properly formatted as a flat list
    inclusion_biomarkers = flatten_list(parsed_input.get('inclusion_biomarker', []))
    exclusion_biomarkers = flatten_list(parsed_input.get('exclusion_biomarker', []))

    query_text = f"""
    Biomarkers: {', '.join(inclusion_biomarkers)}
    Exclusions: {', '.join(exclusion_biomarkers)}
    Status: {parsed_input.get('status', '')}
    Study Size: {parsed_input.get('study_size', '')}
    Ages: {parsed_input.get('ages', '')}
    Gender: {parsed_input.get('gender', '')}
    Country: {parsed_input.get('country', '')}
    """

    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)

    # Query ChromaDB for similar records
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20  # Fetch top 20 matches
    )

    # Convert results into a DataFrame
    if results and "metadatas" in results and results["metadatas"]:
        df = pd.DataFrame(results["metadatas"][0])
        df = apply_filters(df, parsed_input)  # 🔥 Apply the dynamic filters here
        return df
    else:
        return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

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
user_input = st.text_area("Enter clinical trial eligibility criteria:", placeholder="e.g., BRAF, age <= 60, country=USA")

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
                # Show only selected columns in a plain table
                display_columns = ["nctId", "condition", "overallstatus", "count", "sex", "startdate", "country"]
                trial_results = trial_results.rename(columns={
                    "nctId": "Trial ID",
                    "condition": "Condition",
                    "overallstatus": "Status",
                    "count": "Study Size",
                    "sex": "Gender",
                    "startdate": "Start Date",
                    "country": "Country"
                })[display_columns]

                # Display as a clean table
                st.table(trial_results)
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

