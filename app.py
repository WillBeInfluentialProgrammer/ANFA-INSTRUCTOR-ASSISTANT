import streamlit as st
import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import glob


# ======================
# CONFIGURATION & SETUP
# ======================

st.set_page_config(page_title="ANFA Instructor Assistant", page_icon="🛡️", layout="centered")

DATA_FOLDER = "data"
CHROMA_PERSIST_DIR = "chroma_db"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 🔐 Use environment variable in production; fallback for demo only
GROQ_API_KEY = os.getenv('GROQ_API_KEY') 

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()


# ======================
# DATA LOADING & EMBEDDING
# ======================

def data_loader(folder_path: str = DATA_FOLDER) -> list:
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        st.warning(f"No PDFs found in {folder_path}. Please add some PDF files.")
        return []
    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        # Optional: Add source metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf_file)
        all_documents.extend(docs)
    return all_documents


# ======================
# VECTORSTORE MANAGEMENT
# ======================

@st.cache_resource
def get_or_create_vectorstore():
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_model
        )
    else:
        #st.info("🆕 Creating new vectorstore from PDFs...")
        documents = data_loader(DATA_FOLDER)
        if not documents:
            st.error("No documents loaded. Cannot create vectorstore.")
            return None

        # 📏 Improved chunking for legal text (more context retention)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        
)


        document_chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
        vectorstore.persist()
        #st.success(f"✅ Vectorstore created with {len(document_chunks)} chunks.")
    return vectorstore


# ======================
# QA CHAIN SETUP
# ======================
@st.cache_resource
def create_qa_chain(_vectorstore):
    if _vectorstore is None:
        return None

    groq_client = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_prompt = ChatPromptTemplate.from_template(
    """
    You are ANFA, an expert instructor assistant for legal and regulatory training.
    Your task is to help instructors by generating high-quality educational content **strictly based on the provided context**.

  
When asked to generate **questions**, always return the **question text only** — do NOT provide answers unless explicitly requested.

When asked to generate **answers**, provide detailed explanations based ONLY on context.


You may be asked to:
- Answer factual questions → respond with concise, grounded facts
- Generate multiple-choice questions (MCQs) → include 4 options + correct answer + explanation
- Create quizzes → group MCQs or short-answer questions
- Write **descriptive questions** → return clear, exam-style essay questions (NO ANSWERS)
- Design scenario-based assignments → base on real examples from documents
- Formulate short/long questions → write prompts suitable for exams


    🔒 RULES:
    1. **All content MUST be derived ONLY from the context below.**
    2. **Do NOT use external knowledge, even if you know the topic.**
    3. If the context lacks sufficient information to fulfill the request, respond with:
       "I cannot generate this based on the available documents."
    4. For MCQs:
   - Generate exactly 3–5 MCQs unless specified otherwise.
   - Format each MCQ like this:
     ```
     [Number]. [Question text]

        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]

        ✅ Correct Answer: [Letter]  
        📝 Explanation: [Brief explanation from context]
     ```
   - Use clear spacing and indentation for readability.
   - Never merge options into one line.
   - Always include explanation from context.
    
    5. For scenarios: Base them on real examples or rules mentioned in the documents.
    6. Keep language professional, clear, and suitable for training.

    Context:
    {context}

    Previous conversation:
    {chat_history}

    Question/Instruction:
    {question}

    Response:
    """
)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_client,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt
        },
        memory=None  # We manage history manually via session_state
    )

    return qa_chain



# ======================
# SESSION STATE
# ======================
if 'history' not in st.session_state:
    st.session_state['history'] = []


# ======================
# LOAD RESOURCES
# ======================
vectorstore = get_or_create_vectorstore()
qa_chain = create_qa_chain(vectorstore)

if qa_chain is None:
    st.stop()


# ======================
# STREAMLIT UI (Modern Chat)
# ======================

st.title("🛡️ ANFA  INSTRUCTOR'S ASSISTANT")

# Display chat history using chat_message
for entry in st.session_state['history']:
    with st.chat_message("user"):
        st.markdown(entry['question'])
    with st.chat_message("assistant"):
        st.markdown(entry['answer'])
        if entry['sources']:
            with st.expander("📚 Sources"):
                for doc in entry['sources']:
                    source = doc.metadata.get("source", "Unknown source")
                    st.markdown(f"**Source:** `{source}`")
                    st.text(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)

# Chat input with built-in send arrow
if prompt := st.chat_input("💬 Ask Something"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reserve space for assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_container.markdown("🔍 Thinking...")

        try:
            # Build chat history as list of tuples
            chat_history = [
                (entry["question"], entry["answer"])
                for entry in st.session_state["history"]
            ]

            # Run chain
            result = qa_chain({
                "question": prompt,
                "chat_history": chat_history
            })

            answer = result["answer"]
            source_docs = result.get("source_documents", [])

            # Replace placeholder with real answer
            response_container.markdown(answer)

            # Show sources if any
            if source_docs:
                with st.expander("📚 Sources"):
                    for doc in source_docs:
                        source = doc.metadata.get("source", "Unknown")
                        content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
                        st.markdown(f"**Source:** `{source}`")
                        st.text(content)

            # Save to history
            st.session_state["history"].append({
                "question": prompt,
                "answer": answer,
                "sources": source_docs
            })

        except Exception as e:
            response_container.error(f"❌ Error: {str(e)}")

 

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("⚙️ System Info")
    st.markdown(f"**Embedding Model**: `all-MiniLM-L6-v2`")
    st.markdown(f"**LLM**: `llama-3.1-8b-instant` (Groq)")
    st.markdown(f"**Chunk Size**: `1200` | **Overlap**: `200`")
    st.markdown(f"**Top K Docs**: `5`")

    if st.button("🔄 Rebuild Vectorstore"):
        st.cache_resource.clear()
        st.rerun()