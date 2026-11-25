from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# Embeddings
embeddings = MistralAIEmbeddings()

# Vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Base retriever
retriever = vectorstore.as_retriever()

# LLM
llm = ChatMistralAI(model="mistral-small-latest")

# MultiQuery Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

documents = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")

for doc in documents:
    print(doc.page_content)