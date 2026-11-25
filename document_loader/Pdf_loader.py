from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    # separator=" ",
    chunk_size = 1000,
    chunk_overlap = 1
)

data = PyPDFLoader("document_loader/GRU.pdf")
docs = data.load()
chunks = splitter.split_documents(docs)
# print(len(docs))

# print(docs[6])

# print(len(chunks))
print(chunks[0].page_content)

