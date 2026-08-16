import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


VECTOR_DB_PATH = "faiss_welfare"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def get_embedding_model():
    """한국어 특화 임베딩 모델을 생성합니다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_db():
    """저장해 둔 FAISS 벡터 DB를 불러옵니다."""
    embedding_model = get_embedding_model()

    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(
            f"'{VECTOR_DB_PATH}' 폴더를 찾을 수 없습니다. "
            "먼저 make_vector_db.py를 실행해 벡터 DB를 만들어 주세요."
        )

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def format_source(metadata):
    """검색된 문서의 출처를 읽기 쉬운 형태로 표시합니다."""
    source = metadata.get("source", "출처 미상")

    if metadata.get("type") == "pdf":
        page = metadata.get("page_number", "?")
        return f"{source} / {page}쪽"

    return source


def shorten_text(text, max_length=160):
    """참고 자료 미리보기를 짧게 표시합니다."""
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def create_answer_chain():
    """RAG 답변 생성용 LangChain 체인을 만듭니다."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
너는 장애인복지사업 정보를 안내하는 챗봇이다.
반드시 아래 참고 자료에 있는 내용만 사용하여 답변해라.

답변 규칙:
- 질문에 필요한 핵심 내용을 먼저 말한다.
- 참고 자료에 없는 내용은 추측하거나 만들어내지 않는다.
- 자료로 확인할 수 없으면 "제공된 자료에서 확인하기 어렵습니다."라고 답한다.
- 한국어로 자연스럽고 이해하기 쉽게 답한다.
- 복지사업의 대상, 지원 내용, 신청 방법 등의 조건은 자료에 적힌 범위에서 정확하게 설명한다.

[참고 자료]
{context}
            """.strip(),
        ),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


def ask_chatbot(vector_db, answer_chain, question, k=4):
    """관련 문서를 검색하고 GPT 답변을 생성합니다."""
    documents = vector_db.similarity_search(question, k=k)

    context_blocks = []

    for index, document in enumerate(documents, start=1):
        source_text = format_source(document.metadata)

        context_blocks.append(
            f"[자료 {index} | {source_text}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_blocks)

    answer = answer_chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "documents": documents,
    }
