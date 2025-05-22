import streamlit as st
import os
import pdfplumber
import pytesseract
from PIL import Image
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# SET YOUR OPENAI API KEY
openai.api_key = "your-openai-api-key"

# Create folders
os.makedirs("uploaded_docs", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

st.title("Gen-AI Document Chatbot")

# Extract text
def extract_text(file_path, file_type):
    if file_type == "application/pdf":
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

# Embed document
def embed_text(doc_name, text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text], metadatas=[{"source": doc_name}])
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="vector_db")
    vectordb.persist()

# Answer questions
def ask_question(q):
    vectordb = Chroma(persist_directory="vector_db", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
    return qa.run(q)

# Upload
uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded!")

    file_text = extract_text(file_path, uploaded_file.type)
    if file_text:
        st.success("Text extracted.")
        embed_text(uploaded_file.name, file_text)
        st.success("Document embedded.")

# Query
query = st.text_input("Enter your question:")
if query:
    answer = ask_question(query)
    st.subheader("Answer:")
    st.write(answer)
