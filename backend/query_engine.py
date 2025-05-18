import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
from langchain.prompts import PromptTemplate  
import os
from dotenv import load_dotenv

#for api key
load_dotenv()

#Globals
index = None
doc_mapping = None
embedder = None
llm = None
is_initialized = False

def initialize_engine():
    """Load resources only once."""
    global index, doc_mapping, embedder, llm, is_initialized
    if is_initialized:
        return  # Skip reinitialization

    print(" Initializing query engine...")

    # Step 1: Load and parse the Gale Encyclopedia PDF
    loader = PyPDFLoader("X://Downloads/gale.pdf")
    pages = loader.load()

    # Step 2: Chunk the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(pages)

    # Step 3: Extract chunked text
    chunk_texts = [chunk.page_content for chunk in split_docs]

    # Step 4: Load SentenceTransformer model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Step 5: Create embeddings
    embeddings = np.array([embedder.encode(text) for text in chunk_texts], dtype="float32")

    # Step 6: Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Step 7: Store chunk mapping
    doc_mapping = {i: chunk_texts[i] for i in range(len(chunk_texts))}

    # Step 8: Authenticate and initialize LLM
    api_key = os.getenv('hugging_face_api_key')
    hf_token = api_key
    login(hf_token)

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=hf_token,
        temperature=0.5,
        max_new_tokens=200,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    is_initialized = True
    print(f" Engine initialized with {len(chunk_texts)} chunks.")

def retrieve_relevant_docs(query, k=3):
    """Retrieve top-k most relevant medical documents using FAISS."""
    initialize_engine()
    query_embedding = np.array([embedder.encode(query)], dtype="float32")
    distances, indices = index.search(query_embedding, k)
    return [doc_mapping[idx] for idx in indices[0] if idx in doc_mapping]

def query_mediquery(question):
    """Retrieve relevant medical info & generate an AI response."""
    relevant_docs = retrieve_relevant_docs(question, k=2)
    context = " ".join(relevant_docs) if relevant_docs else "No relevant medical info found."

    prompt = f"""You are a knowledgeable and concise medical assistant. Based **only** on the context below, provide a clear, focused, and medically accurate answer to the user's question.

Context:
{context}

Question: {question}
Answer:"""

    return llm.invoke(prompt).strip()
