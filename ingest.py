from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

documents = []

# holidays
loader1 = TextLoader("faculty.txt")
documents.extend(loader1.load())

# hostel (ADD THIS)
loader2 = TextLoader("hostel.txt")
documents.extend(loader2.load())


embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

