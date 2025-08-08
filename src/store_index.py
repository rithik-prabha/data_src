import os
from src.helper import (
    load_text_documents,
    clean_whitespace,
    remove_noise,
    deduplicate_docs,
    text_split
)
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings



load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ✅ Update this to the absolute or relative path to your text docs folder
FOLDER_PATH = r"/home/sparkout/projects/sparkout-website-chatbot/app/services/links_unique.py"
print("📄 Loading documents...")
docs = load_text_documents(FOLDER_PATH)

print("🧼 Cleaning whitespace...")
docs = clean_whitespace(docs)

print("🔍 Removing noise...")
docs = remove_noise(docs)

print("🧹 Deduplicating documents...")
docs = deduplicate_docs(docs)

print("✂️ Splitting text into chunks...")
text_chunks = text_split(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}  # Change to "cuda" if using GPU
)
# # 👉 You can now pass `chunks` to a vector store (e.g., FAISS, Chroma, etc.)
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "sparkout-chatbot_data"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)



