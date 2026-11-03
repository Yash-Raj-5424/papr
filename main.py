from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("documents/kafka-paper.pdf")
documents = loader.load()

print(f'\nTotal pages in the document: {len(documents)}')

for doc in documents:
    print(doc.page_content[:300])
    print("Metadata: ", doc.metadata)
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
    print('Metadata: ', chunk.metadata)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks[:3]]
)

print("No. of embeddings: ", len(vectors))
print("Embedding dimensions: ", len(vectors[0]))
print("First embedding: ", vectors[0][:10])


vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Total vectors stored: ", vector_store.index.ntotal)

query = "what's the main idea in this doc"

response = vector_store.similarity_search(query, k=3)

for i, res in enumerate(response):
    print(f'\nResponse {i+1}:')
    print(res.page_content)