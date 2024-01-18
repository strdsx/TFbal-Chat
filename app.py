import os
import streamlit as st
from time import sleep

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import ConfigurableField
from operator import itemgetter
from PIL import Image
import templates

st.set_page_config(page_title="TF 고민상담", page_icon="🤔", layout="wide")

hide_css = """
<style>
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

@st.cache_data
def avatar_load(t_path="./images/t_avatar.png", f_path="./images/f_avatar.png"):
    return (Image.open(t_path).resize((256,256)),
            Image.open(f_path).resize((256,256)))

# ----- Sidebar
with st.sidebar:
    openai_api_key = st.text_input(label='**OpenAI API Key**', placeholder='sk-...', type='password')
    model_disabled = True
    if openai_api_key == "":
        st.info('OpenAI API Key를 입력 해주세요.', icon='😀')
    elif 'sk-' not in openai_api_key[:4]:
        st.warning('OpenAI API Key 형식이 잘못 되었습니다.', icon='😅')
    else:
        model_disabled = False

    model_name = st.selectbox(
        label="**Model**",
        options=('gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-1106', 'gpt-4', 'gpt-4-1106-preview'),
        index=0,
        disabled=model_disabled
        )
    # temper = st.number_input(label="**Temperature**", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key='temperature')
    # memory = st.number_input(label='**Memory**', min_value=0, max_value=10, value=5, key='memory')

t_avatar, f_avatar = avatar_load()
st.title("🤔 TF 고민 상담 🤔")
with st.expander(label="**설명** 👇", expanded=False):
    tava, fava = st.columns(2)
    tava.image(t_avatar.resize((128,128)), caption='T-bal')
    fava.image(f_avatar.resize((128,128)), caption='F-bal')

    tava.markdown("""- T-bal: MBTI :red[**T 100%**] 챗봇 입니다.
- 현실적인 해결책을 제시하는 것이 제일 중요합니다.
- 항상 이성적, 논리적, 분석적으로 객관적 사실에 기반하여 답변합니다.
- 상대방의 감정에 쉽게 공감하지 않습니다.
""")
    fava.markdown("""- F-bal: MBTI :blue[**F 100%**] 챗봇 입니다.
- 현실적인 해결책을 제시하는 것은 중요하지 않습니다.
- 논리/객관적이지 않아도, 상대방의 기분을 상하지 않게 답변하는게 제일 중요합니다.
- 상대방의 감정에 공감을 잘 합니다.
""")
    st.divider()
    st.markdown("""⚠️ **주의 사항**
- T-bal, F-bal 모두 이전 대화 내역을 기억하지 못합니다.
- 따라서 항상 완성된 독립형 메세지를 보내야 합니다.
- 대화 내역은 최대 30개 까지만 표출되며, 30개 초과시 오래된 대화 내역 먼저 삭제됩니다.
""")

query = st.chat_input("Send message...", disabled=model_disabled)

if model_disabled:
    st.warning("OpenAI API Key를 확인해 주세요.", icon='✋')
    st.stop()

# ----- Chains
temper=0.7
memory=30

parser = StrOutputParser()
llm = ChatOpenAI(model=model_name, temperature=temper, openai_api_key=openai_api_key)
t_chain = PromptTemplate.from_template(templates.T_TEMPLATE) | llm | parser
f_chain = PromptTemplate.from_template(templates.F_TEMPLATE) | llm | parser

# ----- Chat history
if 't_history' not in st.session_state:
    st.session_state.t_history = []
else:
    if len(st.session_state.t_history) > memory:
        st.session_state.t_history = st.session_state.t_history[-memory:]

if 'f_history' not in st.session_state:
    st.session_state.f_history = []
else:
    if len(st.session_state.f_history) > memory:
        st.session_state.f_history = st.session_state.f_history[-memory:]


with st.container(border=False):
    # Write chat history
    if len(st.session_state.t_history) and len(st.session_state.f_history):
        for (q, ta), (_, fa) in zip(st.session_state.t_history, st.session_state.f_history):
            with st.chat_message("user"):
                st.markdown(q)

            tcol_, fcol_ = st.columns(2)
            with tcol_:
                with st.chat_message("ai", avatar=t_avatar):
                    st.markdown(ta)

            with fcol_:
                with st.chat_message("ai", avatar=f_avatar):
                    st.markdown(fa)

    # Write recent chat
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        tcol, fcol = st.columns(2)
        with tcol:
            with st.chat_message("ai", avatar=t_avatar):
                t_answer = ""
                t_placeholder = st.empty()
                for tt in t_chain.stream({"question": query}):
                    t_answer += tt
                    t_placeholder.markdown(t_answer + "▌")
                    sleep(0.005)
                t_placeholder.markdown(t_answer)
                st.session_state.t_history.append((query, t_answer))

        with fcol:
            with st.chat_message("ai", avatar=f_avatar):
                f_answer = ""
                f_placeholder = st.empty()
                for ft in f_chain.stream({"question": query}):
                    f_answer += ft
                    f_placeholder.markdown(f_answer)
                    sleep(0.005)
                f_placeholder.markdown(f_answer)
                st.session_state.f_history.append((query, f_answer))

