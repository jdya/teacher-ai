import streamlit as st

st.set_page_config(page_title="Streamlit 실습 페이지", page_icon="🧪", layout="centered")

# 1. 제목
st.title("Streamlit 실습 페이지")

# 2. 소제목
st.header("기본 위젯 테스트")

# 3. 체크박스
agree = st.checkbox("이 항목에 동의합니다.")

# 4. 텍스트 입력
name = st.text_input("당신의 이름을 입력하세요.")

# 5. 버튼
submitted = st.button("제출")

# 간단한 제출 처리
if submitted:
    if name.strip():
        st.success(f"{name}님, 제출되었습니다.")
    else:
        st.warning("이름을 입력하세요.")
    st.info(f"동의 여부: {'동의' if agree else '비동의'}")