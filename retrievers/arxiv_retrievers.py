from langchain_community.retrievers import ArxivRetriever

# create the retriever

retriver = ArxivRetriever(
    load_max_docs= 2,
    load_all_available_meta= True
)

# query arxiv
docs = retriver.invoke("langChain")

# print the results
for i, doc in enumerate(docs):

    print(f"\nDocument {i+1}")

    print("Metadata:", doc.metadata)

    print("Title:", doc.metadata.get("title", "Not available"))

    print("Summary:", doc.page_content[:500])