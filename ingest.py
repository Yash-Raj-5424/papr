from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("documents/kafka-paper.pdf")
documents = loader.load()

print(f'\nTotal pages in the document: {len(documents)}')

for doc in documents:
    print(doc.page_content[:300])
    print("-" *50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f'Total chunks: {len(chunks)}')

for i, chunk in enumerate(chunks[:3]):
    print(f'\nChunk {i+1}:')
    print(chunk.page_content)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Total vectors stored: ", vector_store.index.ntotal)

vector_store.save_local("faiss_index")

print("PDF ingestion complete")