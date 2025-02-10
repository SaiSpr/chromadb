# import streamlit as st
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# configuration = {
#     "client": "PersistentClient",
#     "path": "/tmp/.chroma"
# }

# collection_name = "documents_collection"

# conn = st.connection("chromadb",
#                      type=ChromaDBConnection,
#                      **configuration)
# documents_collection_df = conn.get_collection_data(collection_name)
# st.dataframe(documents_collection_df)

import streamlit as st
from gradio_client import Client

# Initialize the Hugging Face client
client = Client("SaiPrakashTut/Galileo_twostep_gpu")

def get_model_response(input_text):
    """Send input text to the Hugging Face model and return the response."""
    try:
        result = client.predict(
            input_text=input_text,
            api_name="/extract_criteria"
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="🧬 Biomarker Extraction Tool 🏥", page_icon="🧬", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 Biomarker Extraction Tool 🏥</h1>
    <p style='text-align: center; font-size: 18px;'>Extract genomic biomarkers from clinical trial texts!</p>
    <hr>
    """, unsafe_allow_html=True)

# User input
user_input = st.text_area("Enter the clinical trial eligibility criteria below:", placeholder="e.g., braf or kras")

if st.button("🔍 Extract Biomarkers"):
    if user_input.strip():
        st.markdown("### 🧬 Extracted Biomarkers:")
        response = get_model_response(user_input)
        st.json(response)
    else:
        st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

