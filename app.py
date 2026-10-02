import streamlit as st

from services.pdf_processor import PDFProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import FAISSVectorStore
from services.groq_service import GroqService
from services.rag_service import RAGService


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

def initialize_session_state():
    """Initialize all required Streamlit session variables."""

    defaults = {
        "documents": [],
        "chunks": [],
        "vector_store": None,
        "knowledge_base_ready": False,
        "chat_history": [],
        "stats": {},
        "embedding_service": None,
        "groq_service": None,
        "rag_service": None,
        "selected_model": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ---------------------------------------------------------
# CACHED SERVICES
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_service(model_name):
    """Load and cache the Sentence Transformer model."""
    return EmbeddingService(model_name)


@st.cache_resource(show_spinner=False)
def load_groq_service():
    """Create and cache the Groq service."""
    return GroqService()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📚 PDF RAG Assistant")

st.markdown(
    """
    Upload one or more PDF documents and ask questions about
    their contents using **Retrieval-Augmented Generation (RAG)**.

    The application extracts PDF text, creates embeddings,
    stores them in FAISS, retrieves relevant chunks, and sends
    only the retrieved context to the Groq LLM.
    """
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.subheader("Chunking Configuration")

    chunk_preset = st.selectbox(
        "Choose configuration",
        [
            "Configuration A - 500 / 50",
            "Configuration B - 1000 / 150",
            "Custom"
        ]
    )

    if chunk_preset == "Configuration A - 500 / 50":

        chunk_size = 500
        chunk_overlap = 50

    elif chunk_preset == "Configuration B - 1000 / 150":

        chunk_size = 1000
        chunk_overlap = 150

    else:

        chunk_size = st.slider(
            "Chunk size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100
        )

        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=150,
            step=25
        )

        if chunk_overlap >= chunk_size:

            st.warning(
                "Chunk overlap must be smaller than chunk size."
            )

    st.info(
        f"Chunk size: **{chunk_size}**  \n"
        f"Chunk overlap: **{chunk_overlap}**"
    )

    top_k = st.slider(
        "Top-K retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    similarity_threshold = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "Lower values retrieve more results. "
            "Higher values are stricter."
        )
    )

    st.divider()

    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.divider()

    build_button = st.button(
        "🔨 Build Knowledge Base",
        use_container_width=True,
        type="primary"
    )

    clear_chat_button = st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    )

    clear_kb_button = st.button(
        "♻️ Clear Knowledge Base",
        use_container_width=True
    )


# ---------------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------------

if clear_chat_button:

    st.session_state.chat_history = []

    st.success("Chat history cleared.")

    st.rerun()


# ---------------------------------------------------------
# CLEAR KNOWLEDGE BASE
# ---------------------------------------------------------

if clear_kb_button:

    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.vector_store = None
    st.session_state.knowledge_base_ready = False
    st.session_state.stats = {}
    st.session_state.chat_history = []

    st.success("Knowledge base cleared.")

    st.rerun()


# ---------------------------------------------------------
# DISPLAY UPLOADED FILES
# ---------------------------------------------------------

if uploaded_files:

    st.subheader("📄 Uploaded Documents")

    cols = st.columns(
        min(len(uploaded_files), 4)
    )

    for index, uploaded_file in enumerate(uploaded_files):

        with cols[index % len(cols)]:

            st.info(
                f"📄 **{uploaded_file.name}**\n\n"
                f"{uploaded_file.size / 1024:.1f} KB"
            )


# ---------------------------------------------------------
# BUILD KNOWLEDGE BASE
# ---------------------------------------------------------

if build_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF first."
        )

    elif chunk_overlap >= chunk_size:

        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

    else:

        try:

            with st.spinner(
                "Processing PDFs and building knowledge base..."
            ):

                # -----------------------------
                # PDF PROCESSING
                # -----------------------------

                pdf_processor = PDFProcessor()

                documents = pdf_processor.process_uploaded_files(
                    uploaded_files
                )

                if not documents:

                    raise ValueError(
                        "No readable text was found in "
                        "the uploaded PDF files."
                    )


                # -----------------------------
                # CHUNKING
                # -----------------------------

                chunks = pdf_processor.create_chunks(
                    documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                if not chunks:

                    raise ValueError(
                        "No chunks were created."
                    )


                # -----------------------------
                # EMBEDDINGS
                # -----------------------------

                embedding_service = load_embedding_service(
                    embedding_model
                )

                embeddings = embedding_service.embed_documents(
                    chunks
                )


                # -----------------------------
                # FAISS
                # -----------------------------

                vector_store = FAISSVectorStore()

                vector_store.build(
                    embeddings,
                    chunks
                )


                # -----------------------------
                # GROQ
                # -----------------------------

                groq_service = load_groq_service()

                selected_model = (
                    groq_service.get_selected_model()
                )


                # -----------------------------
                # SAVE SESSION STATE
                # -----------------------------

                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.embedding_service = (
                    embedding_service
                )
                st.session_state.groq_service = groq_service

                st.session_state.rag_service = RAGService(
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                    groq_service=groq_service
                )

                st.session_state.selected_model = (
                    selected_model
                )

                st.session_state.knowledge_base_ready = True

                pages = len({
                    (
                        doc.metadata.get("source"),
                        doc.metadata.get("page")
                    )
                    for doc in documents
                })

                dimensions = (
                    embedding_service.get_embedding_dimension()
                )

                st.session_state.stats = {
                    "documents": len(uploaded_files),
                    "pages": pages,
                    "chunks": len(chunks),
                    "embedding_model": embedding_model,
                    "embedding_dimension": dimensions,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "model": selected_model
                }

            st.success(
                "✅ Knowledge base created successfully!"
            )

        except Exception as error:

            st.error(
                f"Could not build the knowledge base: {error}"
            )


# ---------------------------------------------------------
# KNOWLEDGE BASE STATUS
# ---------------------------------------------------------

if st.session_state.knowledge_base_ready:

    stats = st.session_state.stats

    st.subheader("🟢 Knowledge Base Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Documents",
            stats.get("documents", 0)
        )

    with col2:
        st.metric(
            "Pages",
            stats.get("pages", 0)
        )

    with col3:
        st.metric(
            "Chunks",
            stats.get("chunks", 0)
        )

    with col4:
        st.metric(
            "Embedding Dimension",
            stats.get("embedding_dimension", 0)
        )

    st.caption(
        f"Embedding model: `{stats.get('embedding_model')}`"
    )

    st.caption(
        f"FAISS vector database: **Ready**"
    )

    st.caption(
        f"Groq model: `{stats.get('model')}`"
    )

    st.caption(
        f"Chunking: {stats.get('chunk_size')} / "
        f"{stats.get('chunk_overlap')}"
    )


# ---------------------------------------------------------
# DOCUMENT INFORMATION
# ---------------------------------------------------------

if st.session_state.documents:

    with st.expander("📑 Processed Document Information"):

        document_summary = {}

        for document in st.session_state.documents:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            document_summary.setdefault(
                source,
                []
            ).append(page)

        for source, pages in document_summary.items():

            st.write(
                f"**{source}** — "
                f"{len(set(pages))} page(s)"
            )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

st.subheader("💬 Ask Questions")

if not st.session_state.knowledge_base_ready:

    st.info(
        "Upload PDF documents and click "
        "**Build Knowledge Base** before asking questions."
    )


# Display previous messages.

for message in st.session_state.chat_history:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )

        if role == "assistant":

            sources = message.get(
                "sources",
                []
            )

            retrieved_chunks = message.get(
                "retrieved_chunks",
                []
            )

            if sources:

                st.markdown("### 📚 Sources")

                for source in sources:

                    st.markdown(
                        f"📄 **{source['source']}** — "
                        f"Page **{source['page']}**"
                    )


            if retrieved_chunks:

                with st.expander(
                    "🔍 Retrieved Context"
                ):

                    for index, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        st.markdown(
                            f"**Chunk {index}**"
                        )

                        st.write(
                            f"Source: "
                            f"{chunk['source']}"
                        )

                        st.write(
                            f"Page: "
                            f"{chunk['page']}"
                        )

                        st.write(
                            f"Similarity: "
                            f"{chunk['score']:.4f}"
                        )

                        st.text(
                            chunk["text"]
                        )

                        st.divider()


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded PDFs..."
)


if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.knowledge_base_ready:

        st.warning(
            "Build the knowledge base first."
        )

    else:

        # -----------------------------
        # DISPLAY USER MESSAGE
        # -----------------------------

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)


        # -----------------------------
        # GENERATE RAG ANSWER
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    rag_service = (
                        st.session_state.rag_service
                    )

                    previous_messages = (
                        st.session_state.chat_history[:-1]
                    )

                    result = rag_service.answer_question(
                        question=question,
                        top_k=top_k,
                        similarity_threshold=(
                            similarity_threshold
                        ),
                        conversation_history=(
                            previous_messages
                        )
                    )

                    answer = result["answer"]

                    st.markdown(answer)


                    # -----------------------------
                    # SOURCES
                    # -----------------------------

                    sources = result["sources"]

                    if sources:

                        st.markdown("### 📚 Sources")

                        for source in sources:

                            st.markdown(
                                f"📄 **{source['source']}** — "
                                f"Page **{source['page']}**"
                            )


                    # -----------------------------
                    # RETRIEVED CONTEXT
                    # -----------------------------

                    retrieved_chunks = (
                        result["retrieved_chunks"]
                    )

                    if retrieved_chunks:

                        with st.expander(
                            "🔍 Retrieved Context"
                        ):

                            for index, chunk in enumerate(
                                retrieved_chunks,
                                start=1
                            ):

                                st.markdown(
                                    f"**Chunk {index}**"
                                )

                                st.write(
                                    f"Source: "
                                    f"{chunk['source']}"
                                )

                                st.write(
                                    f"Page: "
                                    f"{chunk['page']}"
                                )

                                st.write(
                                    f"Similarity: "
                                    f"{chunk['score']:.4f}"
                                )

                                st.text(
                                    chunk["text"]
                                )

                                st.divider()


                    # -----------------------------
                    # SAVE ASSISTANT MESSAGE
                    # -----------------------------

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieved_chunks": retrieved_chunks
                    })


                except Exception as error:

                    error_message = (
                        f"⚠️ {error}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "retrieved_chunks": []
                    })