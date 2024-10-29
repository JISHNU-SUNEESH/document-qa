import streamlit as st
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
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
        "Upload Resume (.txt or .md)", type=("txt", "md")
    )
    document=Document(uploaded_file)
    docs=[document]
    jd = st.text_input(
        "Paste Job Description",
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        index=VectorStoreIndex.from_documents(docs)

        query_engine=index.as_query_engine()
        response = query_engine.query(question)

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
        st.write_stream(response)
