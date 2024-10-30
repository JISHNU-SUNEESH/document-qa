import streamlit as st
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
import tempfile 
# Show title and description.
st.title("📄 AI Resume Analyzer")
st.write(
    "Upload the Resume and Job description the APP will tell the suitablity and Resume Score. "
    "To use this app, you need to provide a MistarlAI API key, which you can get [here](https://console.mistral.ai/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
api_key = st.text_input("API Key", type="password")
if not api_key:
    st.info("Please add your  API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    
    llm=ChatMistralAI(model="mistral-large-latest",api_key=api_key)
    Settings.llm=llm



    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload Resume (.pdf)", type=("pdf")
    )
    if uploaded_file:
         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name
    

    
    
    jd = st.text_input(
        "Paste Job Description",
    )

    # Ask the user for a question via `st.text_area`.
    # question = st.text_area(
    #     "Now ask a question about the document!",
    #     placeholder="Can you give me a short summary?",
    #     disabled=not uploaded_file,
    # )

    if uploaded_file and jd  :

        Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        index=VectorStoreIndex.from_documents(documents)

        query_engine=index.as_query_engine()
        # response = query_engine.query(question)
        skill_query = f"Based on the resume, what are the candidate's key skills relevant to this job description: {jd}?"
        suitability_query = f"Based on the resume and job description {jd}, is the candidate suitable for the position? Explain your reasoning."
        summary_query = "Summarize the candidate's strengths and suitability for the position."
        strength_query=f"Based on the resume and job description: {jd}, what is the score for the resume?give the score in numbers for each attribute"

        skill_response = query_engine.query(skill_query)
        suitability_response = query_engine.query(suitability_query)
        summary_response = query_engine.query(summary_query)
        strength_response = query_engine.query(strength_query)

        # Process the uploaded file and question.
        # document = uploaded_file.read().decode()
        # messages = [
        #     {
        #         "role": "user",
        #         "content": f"Here's a document: {document} \n\n---\n\n {question}",
        #     }
        # ]

        # # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     stream=True,
        # )

        # Stream the response to the app using `st.write_stream`.
        ex1=st.expander("Suitability Response")
        ex2=st.expander("Strength of resume")
        # ex3=st.expander()
        # ex4=st.expander()
        ex1.write(suitability_response.response)
        ex2.write(strength_response.response)
