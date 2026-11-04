from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Loaded FAISS index")

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


context = "\n\n".join(doc.page_content for doc in response)

chain = prompt | llm

while True:
    query = input("\nEnter your query (or 'exit'): ")

    if query.lower == "exit":
        print("Exiting !!")
        break

    response = vector_store.similarity_search(query, k=3) 

    contex = "\n\n".join([doc.page_content for doc in response])

    answer = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    print("\nAnswer: ")
    print(answer.content)
    print("\nSources: ")
    for doc in response:
        print(f'PDF: {doc.metadata['source']}')
        print(f'Page: {doc.metadata['page']+1 }')