import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
PDF_FILES = sorted(DATA_DIR.glob("*.pdf"))
VECTOR_DB_PATH = "faiss_welfare"
START_PAGE = 17
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def pdf_to_documents(
    file_path,
    start_page=17,
    chunk_size=500,
    chunk_overlap=80,
):
    """PDF의 지정한 페이지부터 읽고 검색하기 좋은 크기로 나눕니다."""
    loader = PyPDFLoader(file_path)
    all_pages = loader.load()

    start_index = start_page - 1

    if start_index >= len(all_pages):
        raise ValueError(
            f"시작 페이지가 PDF 전체 페이지 수보다 큽니다. "
            f"전체 페이지 수: {len(all_pages)}"
        )

    selected_pages = all_pages[start_index:]
    cleaned_pages = []

    for page in selected_pages:
        text = page.page_content

        if not text or not text.strip():
            continue

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if not text:
            continue

        original_page = page.metadata.get("page", 0) + 1

        cleaned_pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(file_path).name,
                    "type": "pdf",
                    "page": original_page - 1,
                    "page_number": original_page,
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "다.",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(cleaned_pages)
    final_chunks = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        cleaned_lines = []

        for line in chunk.page_content.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        if not cleaned_text:
            continue

        chunk.page_content = cleaned_text
        chunk.metadata["chunk"] = chunk_number
        final_chunks.append(chunk)

    print(
        f"{Path(file_path).name}: "
        f"전체 {len(all_pages)}페이지 중 "
        f"{start_page}페이지부터 사용 → "
        f"{len(cleaned_pages)}개 유효 페이지 → "
        f"{len(final_chunks)}개 문서 조각"
    )

    return final_chunks


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "data 폴더가 없습니다. 프로젝트 폴더에 data 폴더를 만들고 "
            "PDF 파일을 넣어 주세요."
        )

    if not PDF_FILES:
        raise FileNotFoundError(
            "data 폴더에 PDF 파일이 없습니다. PDF 파일을 넣어 주세요."
        )

    print(f"총 {len(PDF_FILES)}개의 PDF 파일을 처리합니다.")
    all_documents = []

    for pdf_path in PDF_FILES:
        print(f"처리 중: {pdf_path.name}")
        documents = pdf_to_documents(
            pdf_path,
            start_page=START_PAGE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("읽을 수 있는 문서가 없습니다.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("모든 문서를 임베딩하고 FAISS를 생성합니다.")
    vector_db = FAISS.from_documents(
        documents=all_documents,
        embedding=embedding_model,
    )

    vector_db.save_local(VECTOR_DB_PATH)

    print(f"완료: '{VECTOR_DB_PATH}' 폴더에 저장했습니다.")
    print(f"전체 PDF 파일 수: {len(PDF_FILES)}")
    print(f"전체 문서 조각 수: {len(all_documents)}")


if __name__ == "__main__":
    main()
