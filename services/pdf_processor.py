import fitz

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class PDFProcessor:
    """Process PDF files and create LangChain documents."""

    def process_uploaded_files(self, uploaded_files):
        """
        Extract text from every uploaded PDF page.

        Each page becomes a Document with metadata containing:
        - source
        - page
        """

        documents = []

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            try:

                pdf_bytes = uploaded_file.getvalue()

                if not pdf_bytes:

                    raise ValueError(
                        f"{filename} is empty."
                    )

                pdf = fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )

            except Exception as error:

                raise ValueError(
                    f"Could not open '{filename}' as a PDF: "
                    f"{error}"
                )


            try:

                if pdf.page_count == 0:

                    raise ValueError(
                        f"'{filename}' contains no pages."
                    )


                readable_pages = 0


                for page_index in range(
                    pdf.page_count
                ):

                    page = pdf.load_page(
                        page_index
                    )

                    text = page.get_text(
                        "text"
                    ).strip()


                    if not text:

                        continue


                    readable_pages += 1


                    document = Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_index + 1
                        }
                    )

                    documents.append(
                        document
                    )


                if readable_pages == 0:

                    raise ValueError(
                        f"No readable text was found "
                        f"in '{filename}'. "
                        "The PDF may contain scanned images "
                        "without an OCR text layer."
                    )

            finally:

                pdf.close()


        return documents


    def create_chunks(
        self,
        documents,
        chunk_size=1000,
        chunk_overlap=150
    ):
        """
        Split page-level documents into smaller chunks.

        Metadata from each original page is preserved.
        """

        if chunk_overlap >= chunk_size:

            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than zero."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(
            documents
        )

        # Add a simple chunk ID.
        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = index

        return chunks