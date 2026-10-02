import faiss
import numpy as np


class FAISSVectorStore:
    """
    Lightweight FAISS vector store.

    Uses normalized embeddings + inner product,
    which corresponds to cosine similarity.
    """

    def __init__(self):

        self.index = None
        self.documents = []


    def build(
        self,
        embeddings,
        documents
    ):

        if embeddings is None:

            raise ValueError(
                "Embeddings cannot be None."
            )

        if len(embeddings) == 0:

            raise ValueError(
                "No embeddings were provided."
            )

        if len(embeddings) != len(documents):

            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )


        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )


        dimension = vectors.shape[1]


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(vectors)

        self.documents = list(
            documents
        )


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        if self.index is None:

            raise RuntimeError(
                "FAISS index has not been built."
            )

        if not self.documents:

            raise RuntimeError(
                "No documents are stored in FAISS."
            )


        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )


        actual_k = min(
            top_k,
            len(self.documents)
        )


        scores, indices = self.index.search(
            query_vector,
            actual_k
        )


        results = []


        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:

                continue


            document = self.documents[
                int(index)
            ]


            results.append({
                "document": document,
                "score": float(score)
            })


        return results


    def size(self):

        if self.index is None:

            return 0

        return self.index.ntotal