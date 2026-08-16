import os

import streamlit as st

from chatbot import (
    ask_chatbot,
    create_answer_chain,
    format_source,
    load_vector_db,
    shorten_text,
)


st.set_page_config(
    page_title="장애인복지사업 정보 챗봇",
    page_icon="♿",
    layout="centered",
)


# --------------------------------------------------
# 기본 화면
# --------------------------------------------------

st.title("♿ 장애인복지사업 정보 챗봇")
st.caption("장애인복지사업 자료를 바탕으로 필요한 정보를 찾아드립니다.")

with st.expander("이 챗봇은 어떻게 답변하나요?"):
    st.write(
        "사용자의 질문과 관련된 복지사업 자료를 먼저 검색한 뒤, "
        "검색된 자료를 참고하여 답변을 생성하는 RAG 방식의 챗봇입니다."
    )
    st.info(
        "챗봇의 답변은 참고용이며, 실제 지원 대상 및 신청 가능 여부는 "
        "반드시 해당 기관의 최신 안내를 확인하세요."
    )


# --------------------------------------------------
# API 키
# --------------------------------------------------

if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "OpenAI API 키가 설정되지 않았습니다. "
        ".streamlit/secrets.toml 파일을 확인하세요."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# --------------------------------------------------
# 무거운 객체는 한 번만 생성
# --------------------------------------------------

@st.cache_resource
def get_resources():
    vector_db = load_vector_db()
    answer_chain = create_answer_chain()
    return vector_db, answer_chain


try:
    vector_db, answer_chain = get_resources()
except Exception as error:
    st.error(f"챗봇을 불러오는 중 오류가 발생했습니다: {error}")
    st.info(
        "처음 실행하는 경우 make_vector_db.py를 먼저 실행하여 "
        "faiss_welfare 폴더를 만들어 주세요."
    )
    st.stop()


# --------------------------------------------------
# 대화 기록
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 참고한 자료"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**{index}. {source['name']}**")
                    st.caption(source["preview"])


# --------------------------------------------------
# 사용자 질문
# --------------------------------------------------

question = st.chat_input("궁금한 장애인복지사업을 질문해 주세요.")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("관련 복지사업 정보를 찾는 중입니다..."):
            try:
                result = ask_chatbot(
                    vector_db,
                    answer_chain,
                    question,
                    k=4,
                )

                answer = result["answer"]
                documents = result["documents"]

                st.markdown(answer)

                sources = []

                for document in documents:
                    sources.append({
                        "name": format_source(document.metadata),
                        "preview": shorten_text(
                            document.page_content,
                            max_length=160,
                        ),
                    })

                with st.expander("📚 참고한 자료"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(f"**{index}. {source['name']}**")
                        st.caption(source["preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                error_message = f"답변을 생성하는 중 오류가 발생했습니다: {error}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })


# --------------------------------------------------
# 사이드바
# --------------------------------------------------

with st.sidebar:
    st.header("챗봇 안내")

    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown(
        """
**사용 예시**
- 장애인연금의 지원 대상은?
- 여성장애인 관련 서비스가 있나요?
- 장애인등록증을 재발급하려면 어떻게 하나요?
- 장애인 정보화 교육의 지원 대상은?
        """
    )
