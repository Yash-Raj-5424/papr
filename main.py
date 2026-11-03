from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/kafka-paper.pdf")
documents = loader.load()

print(f'\nTotal pages in the document: {len(documents)}')

for doc in documents:
    print(doc.page_content[:300])
    print("Metadata: ", doc.metadata)
    print("-" *50)