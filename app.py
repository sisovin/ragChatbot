
import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain


# function to export text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)
  
  # function to load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
  
# function to build the QA chain
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    # Load the QA chain for combining documents
    llm = Ollama(model="llama3.2:3b")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain

st.title("Local RAG Chatbot with PDF")
st.write("Upload a PDF and ask questions based on the content of the PDF")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    text = extract_text_from_pdf(pdf_path)
    
    st.info("Creating FAISS vector store from PDF text. This may take a while...")
    create_faiss_vector_store(text)
    
    st.info("Initializing chatbot...")
    qa_chain = build_qa_chain()
    st.success("Chatbot is ready!")
    
if 'qa_chain' in locals():
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        answer = qa_chain.run(question)
        st.success(answer)
    