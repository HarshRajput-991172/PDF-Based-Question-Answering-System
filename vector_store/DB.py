from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content = "This is the book on AI", metadata = {"source":"AI_book"}),
    Document(page_content = "This is the book on Data Science", metadata = {"source":"datascience_book"}),
    Document(page_content = "This is the book on Deep Learning", metadata = {"source":"DL_book"})
    
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory="chroma_db"
)

result = vectorstore.similarity_search("what is used for data science?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    print()
    print()

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning?")
