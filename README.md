# yamini-wasserstoff-AiInternTask
# app.py

import pytesseract
from PIL import Image
import pdfplumber
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import openai
import pandas as pd

# Make sure you set your OpenAI API key
openai.api_key = "your-openai-key"

# OCR for images
def extract_text_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

# Extract text from PDFs
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Chunk and embed document
def chunk_and_embed(doc_id, raw_text):
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([raw_text], metadatas=[{"source": doc_id}])
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="vector_db")
    vectordb.persist()
    return vectordb

# Ask a question
def get_response(query):
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
    return qa.run(query)

# Identify themes
def extract_themes(responses, n_clusters=2):
    texts = [r["text"] for r in responses]
    doc_ids = [r["doc_id"] for r in responses]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

    themes = []
    for i in range(n_clusters):
        cluster_text = "\n".join([texts[j] for j in range(len(texts)) if kmeans.labels_[j] == i])
        prompt = f"Summarize the following into a research theme:\n{cluster_text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        theme_summary = response.choices[0].message.content.strip()
        cluster_docs = [doc_ids[j] for j in range(len(texts)) if kmeans.labels_[j] == i]
        themes.append(f"Theme {i + 1} ({', '.join(set(cluster_docs))}): {theme_summary}")
    return themes

# Streamlit interface
st.title("Wasserstoff Gen-AI Document Research Bot")
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(path)
    else:
        text = extract_text_from_image(path)

    if text:
        st.success("Document parsed successfully.")
        chunk_and_embed(uploaded_file.name, text)
        st.info("Document embedded for querying.")

query = st.text_input("Enter your research question:")
if query:
    result = get_response(query)
    st.subheader("Query Response")
    st.write(result)
    df = pd.DataFrame({
        "Document ID": ["DOC001"],
        "Extracted Answer": [result],
        "Citation": ["Page 3, Para 2"]  # Placeholder citation
    })
    st.table(df)

    responses = [
        {"doc_id": "DOC001", "text": result},
        {"doc_id": "DOC002", "text": "Company delayed disclosures violating SEBI Act."},
        {"doc_id": "DOC003", "text": "Penalty imposed under Section 15 of SEBI rules."}
    ]
    if st.button("Identify Themes"):
        themes = extract_themes(responses)
        st.subheader("Identified Themes")
        for t in themes:
            st.markdown(t)

