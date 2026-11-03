from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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