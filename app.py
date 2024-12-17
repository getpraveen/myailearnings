import os
import sys
import streamlit as st

#add the current directory to the executable path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pdf_rag import *

EMBEDDING_MODEL = 'nomic-embed-text'
VECTOR_STORE_NAME = 'resume_vector_store'

# Streamlit app title
st.title("Document Upload and LLM Response")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose a document...", type=["pdf"])

if uploaded_file is not None:
    # Load the document
    data = ingest_pdf(uploaded_file)
    
    # generate chunk for the file
    chunks = split_documents(data)

    # Initialize the vector database
    vector_store = create_vector_db(chunks)
    

    # Display uploaded document content
    st.write("Uploaded Document Content:")
    
    # initialize the llm model
    llm_model = get_llm_model()
    
    # initialize the retrieval model
    retriever = create_retriever(vector_store.as_retriver(),llm=llm_model)
    
    chain = create_chain(retriever, llm_model)
    # Text input for the query
    user_query = st.text_input("Enter your query:")

    if st.button("Get Response"):
        # Query the LLM
        response = chain.invoke(user_query)
        st.write("Response from LLM:")
        st.write(response)

# Note: Ensure proper handling and security measures for API keys and sensitive data.
