from prompts.rag_prompt import (
    RAG_SYSTEM_PROMPT,
    build_rag_prompt
)


class RAGService:
    """
    Coordinates:

    Question
        ↓
    Question embedding
        ↓
    FAISS search
        ↓
    Relevant chunks
        ↓
    RAG prompt
        ↓
    Groq
        ↓
    Grounded answer
    """

    FALLBACK_MESSAGE = (
        "I could not find the answer in the uploaded documents."
    )


    def __init__(
        self,
        embedding_service,
        vector_store,
        groq_service
    ):

        self.embedding_service = (
            embedding_service
        )

        self.vector_store = vector_store

        self.groq_service = groq_service


    def _build_search_query(
        self,
        question,
        conversation_history
    ):
        """
        Include limited previous user context to help
        resolve follow-up questions.

        The retrieved PDF context remains the only
        factual source for the final answer.
        """

        previous_user_questions = []

        for message in conversation_history:

            if message.get("role") == "user":

                content = (
                    message.get("content", "")
                    .strip()
                )

                if content:

                    previous_user_questions.append(
                        content
                    )


        recent_questions = (
            previous_user_questions[-2:]
        )


        if not recent_questions:

            return question


        return (
            "Previous user question(s):\n"
            + "\n".join(recent_questions)
            + "\n\nCurrent user question:\n"
            + question
        )


    def retrieve(
        self,
        question,
        top_k=3,
        similarity_threshold=0.25,
        conversation_history=None
    ):

        conversation_history = (
            conversation_history or []
        )


        search_query = self._build_search_query(
            question,
            conversation_history
        )


        query_embedding = (
            self.embedding_service
            .embed_query(search_query)
        )


        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )


        # Apply minimum similarity threshold.
        filtered_results = [
            result
            for result in results
            if result["score"]
            >= similarity_threshold
        ]


        return filtered_results


    @staticmethod
    def _format_context(results):

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            text = document.page_content.strip()


            context_parts.append(
                f"""
[Context {index}]
Source: {source}
Page: {page}

{text}
""".strip()
            )


        return "\n\n".join(
            context_parts
        )


    @staticmethod
    def _build_sources(results):

        sources = []

        seen = set()

        for result in results:

            document = result["document"]

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            key = (
                source,
                page
            )

            if key not in seen:

                seen.add(key)

                sources.append({
                    "source": source,
                    "page": page
                })


        return sources


    @staticmethod
    def _build_retrieved_chunks(results):

        chunks = []

        for result in results:

            document = result["document"]

            chunks.append({
                "source": document.metadata.get(
                    "source",
                    "Unknown"
                ),

                "page": document.metadata.get(
                    "page",
                    "Unknown"
                ),

                "score": result["score"],

                "text": document.page_content
            })


        return chunks


    def answer_question(
        self,
        question,
        top_k=3,
        similarity_threshold=0.25,
        conversation_history=None
    ):

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        results = self.retrieve(
            question=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            conversation_history=(
                conversation_history or []
            )
        )


        # ------------------------------------------------
        # NO RELEVANT CHUNKS
        # ------------------------------------------------

        if not results:

            return {
                "answer": self.FALLBACK_MESSAGE,
                "sources": [],
                "retrieved_chunks": []
            }


        # ------------------------------------------------
        # FORMAT RETRIEVED CONTEXT
        # ------------------------------------------------

        context = self._format_context(
            results
        )


        # ------------------------------------------------
        # BUILD RAG PROMPT
        # ------------------------------------------------

        user_prompt = build_rag_prompt(
            context=context,
            question=question
        )


        # ------------------------------------------------
        # GROQ
        # ------------------------------------------------

        answer = self.groq_service.generate(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1
        )


        # ------------------------------------------------
        # RETURN ANSWER + SOURCES
        # ------------------------------------------------

        return {
            "answer": answer,
            "sources": self._build_sources(
                results
            ),
            "retrieved_chunks": (
                self._build_retrieved_chunks(
                    results
                )
            )
        }