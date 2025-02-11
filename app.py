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
import json
import os
from datetime import datetime

# -------------------------------
# ✅ Initialize Hugging Face Client
# -------------------------------
HF_CLIENT = Client("SaiPrakashTut/Galileo_twostep_gpu")

def get_model_response(input_text):
    """Send input text to HF_CLIENT (Hermes‑FT‑synth) for biomarker extraction."""
    try:
        result = HF_CLIENT.predict(
            input_text=input_text,
            api_name="/extract_criteria"
        )
        return result  # Expected JSON with keys: "inclusion_biomarker", "exclusion_biomarker"
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
# ✅ Helper Functions for Filtering
# -------------------------------
def parse_filter_criteria(filter_value):
    """
    Parses filter criteria (>, >=, <, <=, !=, =) into ChromaDB-supported format.
    Example: ">=50" → ("$gte", 50)
    """
    match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
    if match:
        operator_map = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"}
        op, value = match.groups()
        return operator_map.get(op), int(value)
    return None, None

def canonical_country(country):
    """Convert various representations of a country to a canonical name."""
    if not country:
        return country
    c = country.lower().replace(".", "").replace(" ", "")
    if c in ["us", "usa", "unitedstates", "america"]:
        return "United States"
    return country.title()

def canonical_gender(gender):
    """Convert various representations of gender to canonical values."""
    if not gender:
        return gender
    g = gender.lower().strip()
    if g in ["women", "w", "woman", "female", "f"]:
        return "FEMALE"
    elif g in ["men", "m", "man", "male"]:
        return "MALE"
    return gender.upper()

def canonical_status(status):
    """
    Map synonyms for status into one of the standardized values:
    RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED.
    (In this example, "closed"/"finished"/"done"/"terminated" are mapped to "TERMINATED".)
    """
    if not status:
        return ""
    s = status.lower().strip()
    mapping = {
        "closed": "TERMINATED",
        "finished": "TERMINATED",
        "done": "TERMINATED",
        "terminated": "TERMINATED",
        "recruiting": "RECRUITING",
        "enrolling": "RECRUITING",
        "open": "RECRUITING",
        "withdrawn": "WITHDRAWN",
        "not yet recruiting": "NOT_YET_RECRUITING",
        "active": "ACTIVE_NOT_RECRUITING"
    }
    return mapping.get(s, "UNKNOWN")

def standardize_numeric_filter(filter_str):
    """
    Convert natural language numeric criteria into a symbol-based format.
    E.g., "less than 14" becomes "<14", "greater than or equal to 12" becomes ">=12".
    """
    filter_str = filter_str.lower().strip()
    if "less than or equal to" in filter_str:
        match = re.search(r"less than or equal to\s*(\d+)", filter_str)
        if match:
            return "<=" + match.group(1)
    if "greater than or equal to" in filter_str:
        match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
        if match:
            return ">=" + match.group(1)
    if "less than" in filter_str:
        match = re.search(r"less than\s*(\d+)", filter_str)
        if match:
            return "<" + match.group(1)
    if "greater than" in filter_str:
        match = re.search(r"greater than\s*(\d+)", filter_str)
        if match:
            return ">" + match.group(1)
    match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
    if match:
        op, value = match.groups()
        return op + value
    return filter_str

def standardize_date_filter(filter_str):
    """
    Convert natural language date criteria into a symbol-based ISO format.
    For example, "before March 2015" becomes "<2015-03-01".
    If the date is incomplete (e.g., "2011-12"), pad it to "2011-12-01".
    """
    filter_str = filter_str.lower().strip()
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    if "before" in filter_str:
        match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
        if match:
            month_word, year = match.groups()
            month = months.get(month_word.lower(), "01")
            return "<" + f"{year}-{month}-01"
    if "after" in filter_str:
        match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
        if match:
            month_word, year = match.groups()
            month = months.get(month_word.lower(), "01")
            return ">" + f"{year}-{month}-01"
    # If already in full ISO format (YYYY-MM-DD), use it.
    match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
    if match:
        op, date_val = match.groups()
        return op + date_val
    # If in YYYY-MM format, pad it.
    match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
    if match:
        op, date_val = match.groups()
        return op + date_val + "-01"
    return filter_str

def parse_date_filter(filter_value):
    """
    Parse a date filter string in symbol format (e.g., '<2015-03-01') and return (operator, date string).
    """
    match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})", filter_value)
    if match:
        op, date_val = match.groups()
        return op, date_val
    return None, None

def convert_date_to_int(date_str):
    """
    Converts a date string in YYYY-MM-DD format to an integer in YYYYMMDD format.
    For example, "2020-03-01" becomes 20200301.
    """
    try:
        return int(date_str.replace("-", ""))
    except Exception:
        return None

def build_metadata_filter(parsed_input):
    """
    Constructs a ChromaDB-compatible metadata filter using `$and` for multiple conditions.
    Now includes start_date. This version converts the start_date filter value into an integer
    in YYYYMMDD format, since ChromaDB expects numeric values for comparison.
    """
    filters = []
    if parsed_input.get("country"):
        country_val = canonical_country(parsed_input["country"])
        filters.append({"country": {"$eq": country_val}})
    if parsed_input.get("study_size"):
        operator, value = parse_filter_criteria(parsed_input["study_size"])
        if operator:
            filters.append({"count": {operator: value}})
    if parsed_input.get("ages"):
        operator, value = parse_filter_criteria(parsed_input["ages"])
        if operator:
            filters.append({"age": {operator: value}})
    if parsed_input.get("gender"):
        gender_val = canonical_gender(parsed_input["gender"])
        filters.append({"sex": {"$in": ["ALL", gender_val]}})
    if parsed_input.get("status"):
        status_val = canonical_status(parsed_input["status"])
        filters.append({"overallStatus": {"$eq": status_val}})
    if parsed_input.get("start_date"):
        op, date_val = parse_date_filter(parsed_input["start_date"])
        if op and date_val:
            date_int = convert_date_to_int(date_val)
            if date_int is not None:
                op_map = {"<": "$lt", ">": "$gt", "<=": "$lte", ">=": "$gte"}
                filters.append({"startDate": {op_map.get(op, op): date_int}})
    if len(filters) == 1:
        return filters[0]
    elif len(filters) > 1:
        return {"$and": filters}
    else:
        return None

# -------------------------------
# ✅ OpenAI Structured Filter Extraction Function
# -------------------------------
def test_extract_filters(text):
    """
    Uses OpenAI's function calling to extract filter criteria from the provided text.
    Returns a dict with keys: status, study_size, ages, gender, country, start_date.
    """
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    functions = [
        {
            "name": "extract_filters",
            "description": (
                "Extract filter criteria from clinical trial eligibility text and return a JSON object with keys: "
                "status, study_size, ages, gender, country, start_date. The output should be standardized as follows: "
                "- For 'study_size' and 'ages', return the criteria in symbol-based format (e.g., '<14', '>=12'). "
                "- For 'start_date', return the criteria in symbol-based date format (e.g., '<2015-03-01' means trials starting before March 1, 2015). "
                "- For 'status', the value must be one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED. "
                "If a field is not mentioned, return an empty string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "The clinical trial status in standardized form (e.g., RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED)."
                    },
                    "study_size": {
                        "type": "string",
                        "description": "The study size criteria in symbol-based format (e.g., '<14', '>=12')."
                    },
                    "ages": {
                        "type": "string",
                        "description": "The age criteria in symbol-based format (e.g., '<=65', '>18')."
                    },
                    "gender": {
                        "type": "string",
                        "description": "The gender criteria, e.g., 'male' or 'female'."
                    },
                    "country": {
                        "type": "string",
                        "description": "The country criteria, e.g., 'United States'."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "The start date criteria in symbol-based date format (e.g., '<2015-03-01' means trials starting before March 1, 2015)."
                    }
                },
                "required": ["status", "study_size", "ages", "gender", "country", "start_date"]
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
            {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
        ],
        functions=functions,
        function_call="auto",
        temperature=0.0,
        max_tokens=150,
    )
    message = response["choices"][0]["message"]
    if "function_call" in message:
        arguments = message["function_call"]["arguments"]
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError:
            data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
    else:
        data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
    return data

# -------------------------------
# ✅ Combined Extraction Function
# -------------------------------
def extract_criteria(input_text):
    """
    Splits the input text at the first comma:
      - The text before the comma is sent to HF_CLIENT (Hermes‑FT‑synth) for biomarker extraction.
      - The text after the comma is sent to OpenAI (using structured outputs) for filter extraction.
    Post-processes filter values:
      - For study_size and ages, converts natural language to symbol format.
      - For start_date, converts natural language date expressions to a symbol-based ISO format 
        (padding incomplete dates) and then converts that ISO date to an integer (YYYYMMDD).
      - Canonicalizes status, country, and gender.
    Returns a combined JSON object.
    """
    if ',' in input_text:
        biomarker_text, filter_text = input_text.split(',', 1)
    else:
        biomarker_text = input_text
        filter_text = ""
    
    # Get biomarker extraction output from HF_CLIENT.
    biomarker_data = get_model_response(biomarker_text)
    
    # Extract filter criteria using OpenAI.
    if filter_text.strip():
        filter_data = test_extract_filters(filter_text.strip())
    else:
        filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
    
    # Post-process numeric and date filters.
    filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
    filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
    # Standardize and then convert the start_date to ISO format; then convert to int YYYYMMDD.
    filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
    filter_data["status"] = canonical_status(filter_data.get("status", ""))
    filter_data["country"] = canonical_country(filter_data.get("country", ""))
    filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
    combined = {
        "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
        "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
        "status": filter_data.get("status", ""),
        "study_size": filter_data.get("study_size", ""),
        "ages": filter_data.get("ages", ""),
        "gender": filter_data.get("gender", ""),
        "country": filter_data.get("country", ""),
        "start_date": filter_data.get("start_date", "")
    }
    
    return combined

# -------------------------------
# ✅ Query ChromaDB Based on Combined JSON
# -------------------------------
def flatten_list(nested_list):
    """Flattens a list of lists into a single list."""
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def query_chromadb(parsed_input):
    metadata_filters = build_metadata_filter(parsed_input)
    query_text = f"""
    Biomarkers: {', '.join(flatten_list(parsed_input.get('inclusion_biomarker', [])))}
    Exclusions: {', '.join(flatten_list(parsed_input.get('exclusion_biomarker', [])))}
    Status: {parsed_input.get('status', '')}
    Study Size: {parsed_input.get('study_size', '')}
    Ages: {parsed_input.get('ages', '')}
    Gender: {parsed_input.get('gender', '')}
    Country: {parsed_input.get('country', '')}
    Start Date: {parsed_input.get('start_date', '')}
    """
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20,
        where=metadata_filters
    )
    if results and "metadatas" in results and results["metadatas"]:
        df = pd.DataFrame(results["metadatas"][0])
        return df
    else:
        return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# -------------------------------
# ✅ Format Results as Table
# -------------------------------
def format_results_as_table(df, extracted_data):
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"[{row['nctId']}](https://clinicaltrials.gov/study/{row['nctId']})",
            row["condition"],
            row["overallStatus"],
            row["count"],
            row["age"],
            row["sex"],
            row["startDate"],
            row["country"]
        ])
    table_df = pd.DataFrame(
        table_data,
        columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country"]
    )
    return table_df

# -------------------------------
# ✅ Streamlit UI
# -------------------------------
st.set_page_config(page_title="🧬Galileo", page_icon="🧬", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 Galileo </h1>
    <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
    <hr>
    """, unsafe_allow_html=True)

st.markdown("### 🩸 Enter Biomarker Criteria:")

user_input = st.text_area(
    "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
    placeholder="e.g., 'BRAF, country is us and status terminated with study size less than 40 and start date before March 2020'"
)

if st.button("🔍 Extract Biomarkers & Find Trials"):
    if user_input.strip():
        st.markdown("### 🧬 Extracted Biomarkers & Filters:")
        response = extract_criteria(user_input)
        if isinstance(response, dict):
            st.json(response)  # Display combined JSON output
            st.markdown("### 🔍 Matched Clinical Trials:")
            trial_results = query_chromadb(response)
            if not trial_results.empty:
                formatted_results = format_results_as_table(trial_results, response)
                st.table(formatted_results)
            else:
                st.warning("⚠️ No matching trials found!")
        else:
            st.error("❌ Error in fetching response. Please try again.")
    else:
        st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
    """,
    unsafe_allow_html=True,
)



# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# from gradio_client import Client
# import re
# import json
# import os
# from datetime import datetime

# # -------------------------------
# # ✅ Initialize Hugging Face Client
# # -------------------------------
# HF_CLIENT = Client("SaiPrakashTut/Galileo_twostep_gpu")

# def get_model_response(input_text):
#     """Send input text to Hugging Face model and return the response (biomarker extraction)."""
#     try:
#         result = HF_CLIENT.predict(
#             input_text=input_text,
#             api_name="/extract_criteria"
#         )
#         return result  # Expected JSON with keys: "inclusion_biomarker", "exclusion_biomarker"
#     except Exception as e:
#         return f"Error: {str(e)}"

# # -------------------------------
# # ✅ Initialize ChromaDB
# # -------------------------------
# CHROMA_DB_DIR = "./"  # Ensure correct folder path

# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_collection("clinical_trials")

# # Load embedding model (Force CPU for Hugging Face Spaces)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# # -------------------------------
# # ✅ Helper Functions for Filtering
# # -------------------------------
# def parse_filter_criteria(filter_value):
#     """
#     Parses filter criteria (>, >=, <, <=, !=, =) into ChromaDB-supported format.
#     Example: ">=50" → ("$gte", 50)
#     """
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
#     if match:
#         operator_map = {
#             ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"
#         }
#         op, value = match.groups()
#         return operator_map.get(op), int(value)
#     return None, None

# def canonical_country(country):
#     """Convert various representations of a country to a canonical name."""
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     """Convert various representations of gender to canonical values."""
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     """
#     Map synonyms for status into one of the standardized values:
#     RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED.
#     """
#     if not status:
#         return ""
#     s = status.lower().strip()
#     mapping = {
#         "closed": "TERMINATED",  # Adjust mapping as needed (e.g., closed → TERMINATED or COMPLETED)
#         "finished": "TERMINATED",
#         "done": "TERMINATED",
#         "terminated": "TERMINATED",
#         "recruiting": "RECRUITING",
#         "enrolling": "RECRUITING",
#         "open": "RECRUITING",
#         "withdrawn": "WITHDRAWN",
#         "not yet recruiting": "NOT_YET_RECRUITING",
#         "active": "ACTIVE_NOT_RECRUITING"
#     }
#     return mapping.get(s, "UNKNOWN")

# def standardize_numeric_filter(filter_str):
#     """
#     Convert natural language numeric criteria into a symbol-based format.
#     E.g. "less than 14" becomes "<14", "greater than or equal to 12" becomes ">=12".
#     """
#     filter_str = filter_str.lower().strip()
#     if "less than or equal to" in filter_str:
#         match = re.search(r"less than or equal to\s*(\d+)", filter_str)
#         if match:
#             return "<=" + match.group(1)
#     if "greater than or equal to" in filter_str:
#         match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
#         if match:
#             return ">=" + match.group(1)
#     if "less than" in filter_str:
#         match = re.search(r"less than\s*(\d+)", filter_str)
#         if match:
#             return "<" + match.group(1)
#     if "greater than" in filter_str:
#         match = re.search(r"greater than\s*(\d+)", filter_str)
#         if match:
#             return ">" + match.group(1)
#     # If already in symbol-based format, return as-is.
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
#     if match:
#         op, value = match.groups()
#         return op + value
#     return filter_str

# def standardize_date_filter(filter_str):
#     """
#     Convert natural language date criteria into a symbol-based ISO format.
#     For example, "before March 2015" becomes "<2015-03-01".
#     """
#     filter_str = filter_str.lower().strip()
#     months = {
#         "january": "01", "february": "02", "march": "03", "april": "04",
#         "may": "05", "june": "06", "july": "07", "august": "08",
#         "september": "09", "october": "10", "november": "11", "december": "12"
#     }
#     if "before" in filter_str:
#         match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return "<" + f"{year}-{month}-01"
#     if "after" in filter_str:
#         match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return ">" + f"{year}-{month}-01"
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val
#     return filter_str

# def parse_date_filter(filter_value):
#     """
#     Parse a date filter string in symbol format (e.g., '<2015-03-01') and return (operator, date string).
#     """
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})", filter_value)
#     if match:
#         op, date_val = match.groups()
#         return op, date_val
#     return None, None

# def convert_date_to_timestamp(date_str):
#     """
#     Converts a date string in YYYY-MM-DD format to an integer Unix timestamp.
#     """
#     try:
#         dt = datetime.strptime(date_str, "%Y-%m-%d")
#         return int(dt.timestamp())
#     except Exception as e:
#         return None

# def build_metadata_filter(parsed_input):
#     """
#     Constructs a ChromaDB-compatible metadata filter using `$and` for multiple conditions.
#     Now includes start_date.
#     """
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"age": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if parsed_input.get("start_date"):
#         op, date_val = parse_date_filter(parsed_input["start_date"])
#         if op and date_val:
#             ts = convert_date_to_timestamp(date_val)
#             if ts is not None:
#                 # Map operator to MongoDB-like operator.
#                 op_map = {"<": "$lt", ">": "$gt", "<=": "$lte", ">=": "$gte"}
#                 filters.append({"startDate": {op_map.get(op, op): ts}})
#     if len(filters) == 1:
#         return filters[0]
#     elif len(filters) > 1:
#         return {"$and": filters}
#     else:
#         return None

# # -------------------------------
# # ✅ OpenAI Structured Filter Extraction Function
# # -------------------------------
# def test_extract_filters(text):
#     """
#     Uses OpenAI's function calling to extract filter criteria from the provided text.
#     Returns a dict with keys: status, study_size, ages, gender, country, start_date.
#     """
#     import openai
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": (
#                 "Extract filter criteria from clinical trial eligibility text and return a JSON object with keys: "
#                 "status, study_size, ages, gender, country, start_date. The output should be standardized as follows: "
#                 "- For 'study_size' and 'ages', return the criteria in symbol-based format (e.g., '<14', '>=12'). "
#                 "- For 'start_date', return the criteria in symbol-based date format (e.g., '<2015-03-01' means trials starting before March 1, 2015). "
#                 "- For 'status', the value must be one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED. "
#                 "If a field is not mentioned, return an empty string."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {
#                         "type": "string",
#                         "description": "The clinical trial status in standardized form (e.g., RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, UNKNOWN, ACTIVE_NOT_RECRUITING, COMPLETED)."
#                     },
#                     "study_size": {
#                         "type": "string",
#                         "description": "The study size criteria in symbol-based format (e.g., '<14', '>=12')."
#                     },
#                     "ages": {
#                         "type": "string",
#                         "description": "The age criteria in symbol-based format (e.g., '<=65', '>18')."
#                     },
#                     "gender": {
#                         "type": "string",
#                         "description": "The gender criteria, e.g., 'male' or 'female'."
#                     },
#                     "country": {
#                         "type": "string",
#                         "description": "The country criteria, e.g., 'United States'."
#                     },
#                     "start_date": {
#                         "type": "string",
#                         "description": "The start date criteria in symbol-based date format (e.g., '<2015-03-01' means trials starting before March 1, 2015)."
#                     }
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country", "start_date"]
#             }
#         }
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",  # Adjust model as needed
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
#     return data

# # -------------------------------
# # ✅ Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     """
#     Splits the input text at the first comma:
#       - The text before the comma is sent to HF_CLIENT (Hermes‑FT‑synth) for biomarker extraction.
#       - The text after the comma is sent to OpenAI (using structured outputs) for filter extraction.
#     Post-processes filter values:
#       - For study_size and ages, converts natural language to symbol format.
#       - For start_date, converts natural language date expressions to symbol-based ISO format and then to a Unix timestamp.
#       - Canonicalizes status, country, and gender.
#     Returns a combined JSON object.
#     """
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     # Get biomarker extraction output from HF_CLIENT.
#     biomarker_data = get_model_response(biomarker_text)
    
#     # Extract filter criteria using OpenAI.
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "start_date": ""}
    
#     # Post-process numeric and date filters.
#     filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
#     filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
#     filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
#     filter_data["status"] = canonical_status(filter_data.get("status", ""))
#     filter_data["country"] = canonical_country(filter_data.get("country", ""))
#     filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", ""),
#         "start_date": filter_data.get("start_date", "")
#     }
#     return combined

# # -------------------------------
# # ✅ Query ChromaDB Based on Combined JSON
# # -------------------------------
# def flatten_list(nested_list):
#     """Flattens a list of lists into a single list."""
#     return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

# def query_chromadb(parsed_input):
#     metadata_filters = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Biomarkers: {', '.join(flatten_list(parsed_input.get('inclusion_biomarker', [])))}
#     Exclusions: {', '.join(flatten_list(parsed_input.get('exclusion_biomarker', [])))}
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     Start Date: {parsed_input.get('start_date', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=20,
#         where=metadata_filters
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         return df
#     else:
#         return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# # -------------------------------
# # ✅ Format Results as Table
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             f"[{row['nctId']}](https://clinicaltrials.gov/study/{row['nctId']})",
#             row["condition"],
#             row["overallStatus"],
#             row["count"],
#             row["age"],
#             row["sex"],
#             row["startDate"],
#             row["country"]
#         ])
#     table_df = pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country"]
#     )
#     return table_df

# # -------------------------------
# # ✅ Streamlit UI
# # -------------------------------
# st.set_page_config(page_title="🧬Galileo", page_icon="🧬", layout="wide")

# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 Galileo </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker Criteria:")

# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'BRAF, country is us and status terminated with study size less than 14 and start date before March 2015'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = extract_criteria(user_input)
#         if isinstance(response, dict):
#             st.json(response)  # Display combined JSON output
#             st.markdown("### 🔍 Matched Clinical Trials:")
#             trial_results = query_chromadb(response)
#             if not trial_results.empty:
#                 formatted_results = format_results_as_table(trial_results, response)
#                 st.table(formatted_results)
#             else:
#                 st.warning("⚠️ No matching trials found!")
#         else:
#             st.error("❌ Error in fetching response. Please try again.")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# # -----------------------------------------------------------------------------
# # Footer
# # -----------------------------------------------------------------------------
# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )










# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# from gradio_client import Client
# import re
# import json
# import os

# # -------------------------------
# # ✅ Initialize Hugging Face Client
# # -------------------------------
# HF_CLIENT = Client("SaiPrakashTut/Galileo_twostep_gpu")

# def get_model_response(input_text):
#     """Send input text to Hugging Face model and return the response (biomarker extraction)."""
#     try:
#         result = HF_CLIENT.predict(
#             input_text=input_text,
#             api_name="/extract_criteria"
#         )
#         return result  # Expected to be a JSON object with "inclusion_biomarker" and "exclusion_biomarker"
#     except Exception as e:
#         return f"Error: {str(e)}"

# # -------------------------------
# # ✅ Initialize ChromaDB
# # -------------------------------
# CHROMA_DB_DIR = "./"  # Ensure correct folder path

# # Initialize ChromaDB
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_collection("clinical_trials")

# # Load embedding model (Force CPU for Hugging Face Spaces)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# # -------------------------------
# # ✅ Helper Functions for Filtering (unchanged)
# # -------------------------------

# def parse_filter_criteria(filter_value):
#     """
#     Parses filter criteria (>, >=, <, <=, !=, =) into ChromaDB-supported format.
#     Example: ">=50" → ("$gte", 50)
#     """
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
#     if match:
#         operator_map = {
#             ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"
#         }
#         op, value = match.groups()
#         return operator_map.get(op), int(value)
#     return None, None

# def canonical_country(country):
#     """
#     Maps various country synonyms to a canonical country name.
#     """
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     """
#     Maps various gender synonyms to a canonical gender.
#     """
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     """
#     Maps various status synonyms to a canonical status.
#     """
#     if not status:
#         return status
#     s = status.lower().strip()
#     if s in ["completed", "complete", "finished", "done", "closed"]:
#         return "COMPLETED"
#     elif s in ["recruiting", "enrolling", "open", "open to enrollment", "active recruiting"]:
#         return "RECRUITING"
#     elif s in ["active_not_recruiting", "active not recruiting", "active", "ongoing", "in progress"]:
#         return "ACTIVE_NOT_RECRUITING"
#     else:
#         return status.upper()

# def build_metadata_filter(parsed_input):
#     """
#     Constructs a ChromaDB-compatible metadata filter.
#     """
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"age": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if len(filters) == 1:
#         return filters[0]
#     elif len(filters) > 1:
#         return {"$and": filters}
#     else:
#         return None

# # -------------------------------
# # ✅ Query ChromaDB Based on Combined JSON
# # -------------------------------
# def flatten_list(nested_list):
#     """Flattens a list of lists into a single list."""
#     return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

# def query_chromadb(parsed_input):
#     """Search ChromaDB using the combined JSON (biomarkers + filters)."""
#     metadata_filters = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Biomarkers: {', '.join(flatten_list(parsed_input.get('inclusion_biomarker', [])))}
#     Exclusions: {', '.join(flatten_list(parsed_input.get('exclusion_biomarker', [])))}
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=20,
#         where=metadata_filters
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         return df
#     else:
#         return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# # -------------------------------
# # ✅ Format Results as Table
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     """Format clinical trial results into a structured DataFrame for display."""
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             f"[{row['nctId']}](https://clinicaltrials.gov/study/{row['nctId']})",
#             row["condition"],
#             row["overallStatus"],
#             row["count"],
#             row["age"],
#             row["sex"],
#             row["startDate"],
#             row["country"]
#         ])
#     table_df = pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country"]
#     )
#     return table_df

# # -------------------------------
# # ✅ OpenAI Structured Filter Extraction Function
# # -------------------------------
# def test_extract_filters(text):
#     """
#     Uses OpenAI's function calling to extract filter criteria from the provided text.
#     Returns a dict with keys "status", "study_size", "ages", "gender", and "country".
#     """
#     import openai
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": "Extract filter criteria from clinical trial eligibility text. Return a JSON object with keys: status, study_size, ages, gender, country. If a field is not mentioned, return an empty string.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string", "description": "The clinical trial status, e.g., 'Recruiting', 'Completed', etc."},
#                     "study_size": {"type": "string", "description": "The study size or enrollment count criteria."},
#                     "ages": {"type": "string", "description": "The age criteria, e.g. '>50' or '18-65'."},
#                     "gender": {"type": "string", "description": "The gender criteria, e.g., 'male' or 'female'."},
#                     "country": {"type": "string", "description": "The country criteria."}
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country"]
#             }
#         }
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": ""}
#     return data

# # -------------------------------
# # ✅ Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     """
#     Splits the input text at the first comma.
#       - The text before the comma is processed by the Hermes‑FT‑synth model (biomarkers).
#       - The text after the comma is sent to OpenAI to extract filter criteria.
#     Combines both outputs into one JSON object.
#     """
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     # Get biomarker extraction output from HF_CLIENT
#     biomarker_data = get_model_response(biomarker_text)
#     # If there is text after the comma, get filter criteria via OpenAI; else, use empty defaults.
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": ""}
    
#     # Combine both outputs into one final JSON object
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", "")
#     }
#     return combined

# # -------------------------------
# # ✅ Streamlit UI
# # -------------------------------
# st.set_page_config(page_title="🧬Galileo", page_icon="🧬", layout="wide")

# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 Galileo </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker Criteria:")

# # User Input
# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'BRAF, country:us study_size<33'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         # Use the combined extraction function
#         response = extract_criteria(user_input)
#         if isinstance(response, dict):
#             st.json(response)  # Display the combined JSON output
#             st.markdown("### 🔍 Matched Clinical Trials:")
#             trial_results = query_chromadb(response)
#             if not trial_results.empty:
#                 formatted_results = format_results_as_table(trial_results, response)
#                 st.table(formatted_results)  # Display as static table
#             else:
#                 st.warning("⚠️ No matching trials found!")
#         else:
#             st.error("❌ Error in fetching response. Please try again.")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# # -----------------------------------------------------------------------------
# # Footer
# # -----------------------------------------------------------------------------
# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )




# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# from gradio_client import Client
# import re

# # -------------------------------
# # ✅ Initialize Hugging Face Client
# # -------------------------------
# HF_CLIENT = Client("SaiPrakashTut/Galileo_twostep_gpu")

# def get_model_response(input_text):
#     """Send input text to Hugging Face model and return the response."""
#     try:
#         result = HF_CLIENT.predict(
#             input_text=input_text,
#             api_name="/extract_criteria"
#         )
#         return result  # This should be JSON with extracted biomarkers and filters
#     except Exception as e:
#         return f"Error: {str(e)}"

# # -------------------------------
# # ✅ Initialize ChromaDB
# # -------------------------------
# CHROMA_DB_DIR = "./"  # Ensure correct folder path

# # Initialize ChromaDB
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_collection("clinical_trials")

# # Load embedding model (Force CPU for Hugging Face Spaces)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# # -------------------------------
# # ✅ Helper Functions for Filtering
# # -------------------------------

# def parse_filter_criteria(filter_value):
#     """
#     Parses filter criteria (>, >=, <, <=, !=, =) into ChromaDB-supported format.
#     Example: 
#       ">=50" → ("$gte", 50)
#       "<60" → ("$lt", 60)
#     """
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
#     if match:
#         operator_map = {
#             ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"
#         }
#         op, value = match.groups()
#         return operator_map.get(op), int(value)
#     return None, None  # Return None if no valid filter is found

# # ---
# # New Helper Functions: Canonicalization for country, gender, and status
# # ---

# def canonical_country(country):
#     """
#     Maps various country synonyms to a canonical country name.
#     For example, "us", "u.s", "u.s." "usa", "u.s.a", and "america" will be converted to "United States".
#     """
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     # Add more mappings if needed
#     return country.title()  # Default to title-case

# def canonical_gender(gender):
#     """
#     Maps various gender synonyms to a canonical gender.
#     For example, if the input indicates female ("women", "w", "woman", "female", "f"), it returns "FEMALE".
#     Similarly, male inputs are normalized to "MALE".
#     """
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f", "W", "F"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male", "M"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     """
#     Maps various status synonyms to a canonical status.
#     For example:
#       - Inputs like "complete", "finished", "done" will be mapped to "COMPLETED".
#       - Inputs like "recruiting", "enrolling", "open" will be mapped to "RECRUITING".
#       - Inputs like "active not recruiting", "active", "ongoing", "in progress" will be mapped to "ACTIVE_NOT_RECRUITING".
#     """
#     if not status:
#         return status
#     s = status.lower().strip()
#     if s in ["completed", "complete", "finished", "done", "closed"]:
#         return "COMPLETED"
#     elif s in ["recruiting", "enrolling", "open", "open to enrollment", "active recruiting"]:
#         return "RECRUITING"
#     elif s in ["active_not_recruiting", "active not recruiting", "active", "ongoing", "in progress"]:
#         return "ACTIVE_NOT_RECRUITING"
#     else:
#         # If the input does not match any known synonym, return the uppercase version.
#         return status.upper()

# def build_metadata_filter(parsed_input):
#     """
#     Constructs a ChromaDB-compatible metadata filter using `$and` for multiple conditions.
#     """
#     filters = []

#     # Country Filter (Exact Match using canonicalization)
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})

#     # Study Size Filter (Handles >, >=, <, <=, !=, =)
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})

#     # Age Filter (Handles >, >=, <, <=, !=, =)
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"age": {operator: value}})

#     # Gender Filter (Matches "ALL" or the canonical gender)
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})

#     # Status Filter (Exact Match using canonicalization)
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})

#     # Combining Filters
#     if len(filters) == 1:
#         return filters[0]  # Single filter, no need for `$and`
#     elif len(filters) > 1:
#         return {"$and": filters}  # Apply multiple conditions
#     else:
#         return None  # No filters applied

# # -------------------------------
# # ✅ Query ChromaDB Based on Extracted JSON
# # -------------------------------

# def flatten_list(nested_list):
#     """Flattens a list of lists into a single list."""
#     return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

# def query_chromadb(parsed_input):
#     """Search ChromaDB using extracted biomarker JSON & strict metadata filters."""
#     metadata_filters = build_metadata_filter(parsed_input)

#     query_text = f"""
#     Biomarkers: {', '.join(flatten_list(parsed_input.get('inclusion_biomarker', [])))}
#     Exclusions: {', '.join(flatten_list(parsed_input.get('exclusion_biomarker', [])))}
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     """

#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)

#     # Query ChromaDB with metadata filtering
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=20,  # Fetch top 20 matches
#         where=metadata_filters  # Apply strict filters
#     )

#     # Convert results into a DataFrame
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         return df
#     else:
#         return pd.DataFrame(columns=["nctId", "condition", "eligibility", "briefSummary", "overallStatus", "age", "count", "sex", "country", "startDate"])

# # -------------------------------
# # ✅ Convert Data to Static Table Format
# # -------------------------------
# def format_results_as_table(df, extracted_biomarkers):
#     """Format clinical trial results into a structured DataFrame for display.
#        The Biomarker column is removed and an Ages column is added.
#     """
#     table_data = []
    
#     for _, row in df.iterrows():
#         table_data.append([
#             f"[{row['nctId']}](https://clinicaltrials.gov/study/{row['nctId']})",  # Hyperlinked ID
#             row["condition"],
#             row["overallStatus"],
#             row["count"],
#             row["age"],
#             row["sex"],
#             row["startDate"],
#             row["country"]
#         ])

#     table_df = pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country"]
#     )
    
#     return table_df

# # -------------------------------
# # ✅ Streamlit UI
# # -------------------------------
# st.set_page_config(page_title="🧬Galileo", page_icon="🧬", layout="wide")

# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 Galileo </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker Criteria:")

# # User Input
# user_input = st.text_area("Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#                           placeholder="e.g., Identify lung carer trials for patients with an ALK fusion OR ROS1 rearrangement, age: > 50, gender:male, country:us, study_size:>=50, status=recruiting")

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         # Extract Biomarkers
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = get_model_response(user_input)

#         if isinstance(response, dict):
#             st.json(response)  # Show extracted biomarkers & filters
            
#             # Query ChromaDB with extracted biomarkers
#             st.markdown("### 🔍 Matched Clinical Trials:")
#             trial_results = query_chromadb(response)
            
#             if not trial_results.empty:
#                 formatted_results = format_results_as_table(trial_results, response)
#                 st.table(formatted_results)  # Display as static table
#             else:
#                 st.warning("⚠️ No matching trials found!")
#         else:
#             st.error("❌ Error in fetching response. Please try again.")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# # -----------------------------------------------------------------------------
# # Footer
# # -----------------------------------------------------------------------------
# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )







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

