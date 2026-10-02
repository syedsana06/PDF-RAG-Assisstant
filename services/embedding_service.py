from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Generate embeddings using Sentence Transformers.

    The SAME model is used for:
    - document chunks
    - user questions
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )


    def embed_documents(self, documents):

        if not documents:

            raise ValueError(
                "No documents were provided for embedding."
            )

        texts = [
            document.page_content
            for document in documents
        ]

        try:

            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            return embeddings

        except Exception as error:

            raise RuntimeError(
                f"Document embedding failed: {error}"
            )


    def embed_query(self, query):

        if not query or not query.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        try:

            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            return embedding[0]

        except Exception as error:

            raise RuntimeError(
                f"Question embedding failed: {error}"
            )


    def get_embedding_dimension(self):

        return self.model.get_sentence_embedding_dimension()