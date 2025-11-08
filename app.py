import json
import requests
import streamlit as st

st.set_page_config(page_title="교사용 AI 에이전트 v2", page_icon="🤖", layout="centered")
st.title("교사용 AI 에이전트 v2")


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


# 사이드바: PDF 업로드(임시)
with st.sidebar:
    uploaded_pdf = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    if uploaded_pdf is not None:
        st.success("PDF 업로드 성공! (아직 저장 안 됨)")

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