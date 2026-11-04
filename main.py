from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

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

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3,
    max_output_tokens=256
)

prompt = PromptTemplate.from_template("""
You are a helpful assistant.
Answer teh questions using ONLY the context below.
If the answer is not present in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")

query = "what is the main idea in this document ?"

response = vector_store.similarity_search(query, k=3)

context = "\n\n".join(doc.page_content for doc in response)

chain = prompt | llm

answer = chain.invoke(
    {
        "context": context,
        "question": query
    }
)

print("\nAnswer: ")
print(answer.content)