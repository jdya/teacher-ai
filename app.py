# pip install pypdf2 supabase deepseek
import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from supabase import create_client, Client
try:
    # 전용 DeepSeek Python 라이브러리 (클라이언트 클래스는 api.DeepSeekAPI)
    from deepseek.api import DeepSeekAPI  # type: ignore
except Exception:
    DeepSeekAPI = None  # 라이브러리 미설치 시 None 처리

st.set_page_config(page_title="교사용 AI 에이전트 v3", page_icon="🤖", layout="centered")
st.title("교사용 AI 에이전트 v3")

# Supabase 클라이언트 초기화
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        supabase = None
        st.warning(f"Supabase 초기화 실패: {e}", icon="⚠️")
else:
    st.warning("Supabase 설정(SUPABASE_URL, SUPABASE_KEY)이 없어 저장 기능이 비활성화됩니다.", icon="⚠️")

# DeepSeek 임베딩 모델 설정(요구사항에 맞춰 고정값 사용)
EMBEDDING_MODEL = "deepseek-embed"

# DeepSeek 클라이언트(임베딩용) 초기화
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY")
deepseek_client = None
class _DSResp:
    def __init__(self, data):
        self.data = data

class DeepseekCompatClient:
    """OpenAI 스타일 embeddings.create를 제공하는 간단 래퍼.

    /v1/embeddings 또는 /embeddings를 호출하고,
    404/405 시 임시 1536차원 0 벡터로 폴백합니다.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.embeddings = self.Embeddings(self)

    class Embeddings:
        def __init__(self, parent: "DeepseekCompatClient"):
            self.parent = parent

        def create(self, model: str, input: str):
            payload = {"model": model, "input": input}
            # 1차: /v1/embeddings
            url1 = f"{self.parent.base_url}/v1/embeddings"
            r = requests.post(url1, headers=self.parent.headers, json=payload, timeout=60)
            if r.status_code in (404, 405):
                # 2차: /embeddings
                url2 = f"{self.parent.base_url}/embeddings"
                r2 = requests.post(url2, headers=self.parent.headers, json=payload, timeout=60)
                if r2.status_code < 300:
                    j2 = r2.json()
                    data2 = j2.get("data", [])
                    return _DSResp(data2)
                # 최종 폴백: 임시 벡터 반환
                return _DSResp([{"embedding": [0.0] * 1536}])
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            return _DSResp(data)

if DEEPSEEK_API_KEY:
    try:
        deepseek_client = DeepseekCompatClient(DEEPSEEK_API_KEY)
    except Exception as e:
        deepseek_client = None
        st.warning(f"DeepSeek 클라이언트 초기화 실패: {e}", icon="⚠️")


# 세션 상태 초기화: 대화 기록(messages)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]


def _deepseek_stream(messages: list[dict], api_key: str, model: str = "deepseek-chat"):
    """DeepSeek Chat Completions 스트리밍 응답을 제너레이터로 반환.

    OpenAI 호환 SSE 스트림 형식(data: {json})을 처리합니다.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data_str = raw[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except Exception:
                    continue

                # OpenAI 호환 delta 포맷 처리
                delta = (
                    obj.get("choices", [{}])[0]
                    .get("delta", {})
                )
                content = delta.get("content")
                if content:
                    yield content


def _fallback_stream(prompt: str):
    """API 키가 없거나 오류가 난 경우를 위한 간단한 스트리밍 데모."""
    demo = f"(데모) 입력하신 내용에 대한 응답: {prompt}"
    for ch in demo:
        yield ch


# PDF 텍스트 추출
def get_pdf_text(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")


# DeepSeek 임베딩 생성
def get_embedding(text: str, client) -> list[float]:
    # 요구사항: client.embeddings.create(model="deepseek-embed", ...)
    if client is None:
        raise RuntimeError(
            "DeepSeek 라이브러리가 없거나 클라이언트가 초기화되지 않았습니다. "
            "터미널에서 'pip install deepseek' 실행 후, .streamlit/secrets.toml에 DEEPSEEK_API_KEY를 설정하세요."
        )
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception as e:
        st.sidebar.warning(f"임베딩 호출 실패: {e}. 임시 벡터로 저장합니다.")
        return [0.0] * 1536

    vec = getattr(resp, "data", [None])[0]
    if isinstance(vec, dict):
        vec = vec.get("embedding") or vec.get("vector")

    if not isinstance(vec, list):
        st.sidebar.warning("임베딩 응답 형식 오류: 임시 벡터(0)로 저장합니다.")
        vec = [0.0] * 1536

    # 벡터 길이 정규화(테이블 스키마: 1536차원)
    target_dim = 1536
    if len(vec) > target_dim:
        vec = vec[:target_dim]
    elif len(vec) < target_dim:
        vec = vec + [0.0] * (target_dim - len(vec))
    return vec


# Supabase 저장
def save_to_supabase(content: str, embedding: list[float], file_name: str):
    if not supabase:
        raise RuntimeError("Supabase 클라이언트가 초기화되지 않았습니다.")
    payload = {"content": content, "embedding": embedding, "file_name": file_name}
    res = supabase.table("class_materials").insert(payload).execute()
    # 최신 supabase-py는 .execute()에 data를 포함
    data = getattr(res, "data", None)
    if data is None:
        # 일부 버전은 dict를 반환할 수 있음
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"Supabase 저장 실패: {res['error']}")
    return res


# 파일 목록 가져오기 (최신순)
def fetch_uploaded_files() -> list[dict]:
    """class_materials 테이블에서 file_name과 created_at만 최신순으로 조회."""
    if not supabase:
        raise RuntimeError("Supabase 클라이언트가 초기화되지 않았습니다.")
    try:
        res = (
            supabase
            .table("class_materials")
            .select("file_name, created_at")
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"파일 목록 조회 실패: {e}")

    data = getattr(res, "data", None)
    if data is None and isinstance(res, dict):
        data = res.get("data")
    if not data:
        return []
    return data


# 사이드바: PDF 업로드(임시)
with st.sidebar:
    # 설정 상태 표시: 키/클라이언트 초기화 여부
    st.subheader("설정 상태")
    if DEEPSEEK_API_KEY and deepseek_client is not None:
        st.success("DeepSeek 키 감지 및 클라이언트 준비 완료")
    elif DEEPSEEK_API_KEY and deepseek_client is None:
        st.warning("키는 감지되었지만 클라이언트 초기화에 실패했습니다.")
    else:
        st.error("DeepSeek 키가 설정되지 않았습니다.")

    if supabase:
        st.success("Supabase 연결됨")
    else:
        st.warning("Supabase 비활성화: URL/KEY 확인 필요")

    uploaded_pdf = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    if uploaded_pdf is not None:
        st.caption(f"파일: {uploaded_pdf.name}")
    upload_clicked = st.button("PDF 업로드", disabled=uploaded_pdf is None)
    if upload_clicked and uploaded_pdf is not None:
        # (1) DeepSeek 클라이언트 초기화 확인
        if deepseek_client is None:
            st.error("DeepSeek API 키를 확인하세요.")
        else:
            try:
                # (2) get_pdf_text -> get_embedding -> save_to_supabase 순서 실행
                text = get_pdf_text(uploaded_pdf)
                embedding = get_embedding(text, deepseek_client)
                save_to_supabase(text, embedding, uploaded_pdf.name)
                # (3) 성공 메시지 표시
                st.success(f"{uploaded_pdf.name} 저장 완료!")
                # 저장 성공 후 즉시 새로고침하여 목록에 반영
                st.rerun()
            except Exception as e:
                st.error(f"업로드/저장 실패: {e}")

    # 학습된 파일 목록 표시
    st.sidebar.subheader("학습된 파일 목록")
    try:
        files = fetch_uploaded_files()
        if files:
            st.sidebar.dataframe(files, use_container_width=True, height=240)
        else:
            st.sidebar.caption("아직 저장된 파일이 없습니다.")
    except Exception as e:
        st.sidebar.warning(f"파일 목록을 불러오지 못했습니다: {e}")

# RAG 모드 스위치
rag_mode = st.toggle("🤖 '우리 반 맞춤형' RAG 모드 켜기")

# 기존 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 하단 고정 입력창
prompt = st.chat_input("메시지를 입력하세요…")

if prompt:
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 비밀키 확인 및 DeepSeek 스트리밍 호출
    api_key = st.secrets.get("DEEPSEEK_API_KEY")

    with st.chat_message("assistant"):
        try:
            if rag_mode:
                # 임시 RAG 모드 응답
                def _rag_demo_stream():
                    text = "RAG 모드입니다. (아직 개발 중)"
                    for ch in text:
                        yield ch
                full_text = st.write_stream(_rag_demo_stream())
            else:
                # 일반 모드: DeepSeek API 스트리밍
                if api_key:
                    stream = _deepseek_stream(st.session_state.messages, api_key)
                else:
                    st.info("DEEPSEEK_API_KEY가 설정되지 않아 데모 응답으로 표시합니다.")
                    stream = _fallback_stream(prompt)
                full_text = st.write_stream(stream)
        except Exception as e:
            st.error(f"응답 생성 중 오류가 발생했습니다: {e}")
            full_text = "오류로 인해 응답을 생성하지 못했습니다."

    # 스트리밍이 끝난 후 전체 응답을 세션에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_text})