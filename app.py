import hashlib
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_mistralai.chat_models import MistralAIRateLimitError
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load .env from project root and fallback to Streamlit secrets if present
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

try:
    if "MISTRAL_API_KEY" in st.secrets and not os.getenv("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = str(st.secrets["MISTRAL_API_KEY"])
except Exception:
    pass

mistral_api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")

st.set_page_config(page_title="📚 RAG Chatbot", page_icon="🤖")
st.title("📚 Chat with Your PDF (RAG Bot)")

if not mistral_api_key:
    st.error(
        "MISTRAL_API_KEY not found. Add `MISTRAL_API_KEY=your_key` to your `.env` file "
        "in the project root (or set it in Streamlit Secrets)."
    )
    st.stop()


@st.cache_resource
def get_embeddings(api_key: str) -> MistralAIEmbeddings:
    return MistralAIEmbeddings(api_key=api_key)


@st.cache_resource
def get_llm(api_key: str) -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        max_retries=2,
    )


# ---------------------------
# Session State Initialization
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_file_sig" not in st.session_state:
    st.session_state.processed_file_sig = None

# ---------------------------
# Sidebar - Upload PDF
# ---------------------------
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    file_sig = (uploaded_file.name, len(file_bytes), file_hash)

    # Process only when a new or different PDF is uploaded
    if st.session_state.processed_file_sig != file_sig:
        temp_pdf_path = BASE_DIR / "temp.pdf"
        temp_pdf_path.write_bytes(file_bytes)
        st.sidebar.success("PDF uploaded successfully!")

        persist_dir = str(BASE_DIR / "chroma_db" / f"doc_{file_hash}")

        try:
            loader = PyPDFLoader(str(temp_pdf_path))
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)

            embeddings = get_embeddings(mistral_api_key)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_file_sig = file_sig
            st.sidebar.success("Vector DB created!")

        except MistralAIRateLimitError:
            st.warning(
                "Mistral API rate limit exceeded. Please wait 30–60 seconds and ask your question again."
            )
            st.stop()
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                st.error(
                    "401 Unauthorized: Your MISTRAL_API_KEY is invalid, expired, or inactive. "
                    "Generate a new key from the Mistral Console and update `.env`."
                )
            else:
                st.error(f"Error creating Vector DB: {e}")
            st.stop()
    else:
        st.sidebar.info("Vector DB already ready for this PDF.")

# ---------------------------
# Prompt Template
# ---------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# ---------------------------
# Chat UI
# ---------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---------------------------
# User Input
# ---------------------------
user_input = st.chat_input("Ask something about your document...")

if user_input:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append(HumanMessage(content=user_input))

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": user_input,
        })

        llm = get_llm(mistral_api_key)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                response = llm.invoke(final_prompt)
            except MistralAIRateLimitError:
                time.sleep(2)
                try:
                    response = llm.invoke(final_prompt)
                except MistralAIRateLimitError:
                    st.warning(
                        "Mistral API rate limit exceeded. Please wait 30–60 seconds and ask your question again."
                    )
                    st.stop()
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    st.error(
                        "401 Unauthorized: Your MISTRAL_API_KEY is invalid, expired, or inactive."
                    )
                else:
                    st.error(f"Error generating response: {e}")
                st.stop()

            response_text = response.content
            if isinstance(response_text, list):
                response_text = "".join(
                    item if isinstance(item, str) else item.get("text", "")
                    for item in response_text
                )

            for word in response_text.split():
                full_response += word + " "
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append(AIMessage(content=full_response))

# ---------------------------
# Reset Button
# ---------------------------
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()