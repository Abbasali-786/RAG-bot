
import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="📘 File Q&A with Groq", layout="centered")
st.title("📘 File Question Answering System")
st.markdown("""
Upload a document (PDF, TXT, CSV, DOCX) and ask questions about its content using **LLaMA 3 via Groq**.
""")

@st.cache_resource(show_spinner=True)
def process_file(uploaded_file, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        elif file_type == "csv":
            loader = CSVLoader(file_path)
        elif file_type == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            st.error("Unsupported file type")
            return None

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
        return None
    finally:
        os.remove(file_path)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

def create_chain(llm, vector_store):
    prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Be accurate and concise. If you don't know, say you don't know.

        <context>
        {context}
        </context>

        Question: {input}
    """)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    return create_retrieval_chain(retriever, doc_chain)

uploaded_file = st.file_uploader("📄 Upload your file", type=["pdf", "txt", "csv", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    with st.spinner("Processing file and building knowledge base..."):
        vector_store = process_file(uploaded_file, file_type)

    if vector_store:
        llm = get_llm()
        chain = create_chain(llm, vector_store)

        st.success("✅ File processed. You can now ask questions!")
        user_question = st.text_input("❓ Ask a question about the file")

        if user_question:
            with st.spinner("Generating answer..."):
                try:
                    start = time.time()
                    result = chain.invoke({"input": user_question})
                    elapsed = time.time() - start
                    answer = result['answer'].strip()

                    st.markdown(f"**🧠 Answer (in {elapsed:.2f} sec):**")
                    st.info(answer)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
