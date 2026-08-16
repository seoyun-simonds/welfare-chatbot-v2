# 장애인복지사업 정보 챗봇 — Streamlit 버전

## 1. 폴더 구조

```text
장애인복지챗봇/
├── app.py
├── chatbot.py
├── make_vector_db.py
├── requirements.txt
├── data/
│   └── 장애인복지사업.pdf
├── faiss_welfare/
└── .streamlit/
    └── secrets.toml
```

## 2. 설치

터미널에서 프로젝트 폴더로 이동한 뒤:

```bash
pip install -r requirements.txt
```

## 3. OpenAI API 키 설정

`.streamlit/secrets.toml` 파일을 만들고 다음처럼 입력합니다.

```toml
OPENAI_API_KEY = "여기에_본인의_API_KEY"
```

API 키는 GitHub 등에 공개하지 마세요.

## 4. PDF 넣기

실제 복지사업 PDF를 다음 위치에 넣습니다.

```text
data/장애인복지사업.pdf
```

PDF 파일명이 다르면 `make_vector_db.py`의 `PDF_PATH`를 수정합니다.

## 5. FAISS 벡터 DB 만들기

프로젝트 폴더에서:

```bash
python make_vector_db.py
```

정상적으로 끝나면 `faiss_welfare` 폴더에 벡터 DB가 생성됩니다.

## 6. 웹앱 실행

```bash
streamlit run app.py
```

브라우저에서 Streamlit 페이지가 열립니다.

## 7. 중요한 설정

현재 원본 노트북의 `start_page=17`을 그대로 반영했습니다.

PDF의 17페이지부터 사용하는 것이 의도한 것이 아니라면
`make_vector_db.py`의:

```python
START_PAGE = 17
```

을 `1` 또는 실제 필요한 시작 페이지로 바꾸세요.

## 8. 원본 노트북과 달라진 점

- Google Colab의 `userdata` 대신 `st.secrets` 사용
- `files.upload()` 제거
- `input()` 대신 `st.chat_input()` 사용
- `print()` 대신 Streamlit UI 사용
- FAISS를 웹앱 실행 때마다 만들지 않고 한 번 생성한 뒤 재사용
- 대화 기록을 `st.session_state`로 관리
- 검색된 자료의 출처를 웹 화면에 표시
