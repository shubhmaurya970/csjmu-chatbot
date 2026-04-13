"""
Streamlit frontend with a RAG-backed assistant using LangChain, Chroma, and Groq.
Place a `config.json` beside this file with `GROQ_API_KEY` (and optional `COLLEGE_NAME`).
Run: streamlit run main.py
"""

from datetime import datetime
import json
import os
import re
import time

import streamlit as st
from fpdf import FPDF
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def stream_text(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.02)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def text_for_pdf(text: str) -> str:
    """FPDF core fonts (Helvetica) only support Latin-1; normalize Unicode for export."""
    t = remove_emojis(text or "")
    for old, new in (
        ("\u2014", "-"),  # em dash
        ("\u2013", "-"),  # en dash
        ("\u2012", "-"),  # figure dash
        ("\u2212", "-"),  # minus
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2026", "..."),
        ("\u00a0", " "),
        ("\u2009", " "),
        ("\u200b", ""),
        ("\ufeff", ""),
    ):
        t = t.replace(old, new)
    return t.encode("latin-1", errors="replace").decode("latin-1")


working_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(working_dir, "config.json")
with open(_config_path, encoding="utf-8") as _f:
    config_data = json.load(_f)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

COLLEGE_NAME = config_data.get("COLLEGE_NAME", "Chhatrapati Shahu Ji Maharaj University, Kanpur")
COLLEGE_SHORT = config_data.get("COLLEGE_SHORT", "CSJMU")


def system_prompt_for_college(name: str) -> str:
    return f"""You are a specialized AI assistant for **{name}**. Answer only from the retrieved context and verified university-related information.

**Goals**
1. Help students and visitors with admissions, academics, calendars, programs, and campus services.
2. Be clear, concise, and accurate; use short bullet lists when helpful.
3. Use a warm, professional tone suitable for a public university help desk.

**Rules**
- If the context does not contain the answer, say you do not have that information in the knowledge base and suggest contacting the relevant university office.
- Do not claim to be an official spokesperson or employee; you are an informational assistant.
- Do not give legal, medical, or financial advice beyond general university information.
- Stay on topics related to {name} and the provided documents.

**Formatting Rules**
- Always format answers in clean bullet points
- Use line breaks between points
- Never return long paragraphs
- Use proper markdown formatting for lists, links, and emphasis when appropriate.
"""


def negative_prompt_for_college(name: str) -> str:
    return f"""
- Do not invent policies, dates, fees, or contacts not supported by the context.
- Do not discuss topics unrelated to {name} when the user asked about the university; politely redirect.
- Do not use an unprofessional tone.
"""


DEFAULT_SYSTEM_PROMPT = system_prompt_for_college(COLLEGE_NAME)
DEFAULT_NEGATIVE_PROMPT = negative_prompt_for_college(COLLEGE_NAME)


def contains_sensitive_topics(question):
    if not question:
        return False

    sensitive_keywords = []
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)


@st.cache_resource
def setup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def chat_chain(
    vectorstore,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt_template = f"""{system_prompt}

{negative_prompt}

Context (from the university knowledge base):
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"],
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=False,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


# --- Page & CSJMU-themed UI ---
st.set_page_config(
    page_title=f"{COLLEGE_SHORT} Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Hide Streamlit Cloud Deploy only; use the "⋮" menu (top right) for Settings / theme. */
    .stAppDeployButton,
    .stDeployButton,
    [data-testid="stToolbarDeployButton"] {{
        display: none !important;
    }}
    [data-testid="stChatMessage"] {
    padding: 12px;
    margin-bottom: 10px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.08);
}
    :root {{
        --csjmu-blue: #002D62;
        --csjmu-gold: #C5A028;
        --csjmu-gold-light: #E8D78A;
        --csjmu-surface: #F4F6F9;
    }}
    .block-container {{
        padding-top: 1.25rem;
        max-width: 1100px;
    }}
    /* Follow Streamlit Light/Dark (set in .streamlit/config.toml [theme.light] / [theme.dark]) */
    div[data-testid="stSidebar"] {{
        background: var(--secondary-background-color) !important;
        border-right: 3px solid var(--primary-color);
    }}
    .csjmu-hero {{
        background: linear-gradient(135deg, var(--csjmu-blue) 0%, #0a3d7a 55%, #1a5089 100%);
        color: #fff;
        padding: 1.35rem 1.5rem 1.15rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 45, 98, 0.22);
        border-left: 6px solid var(--csjmu-gold);
    }}
    .csjmu-hero h1 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .csjmu-hero p {{
        margin: 0;
        opacity: 0.95;
        font-size: 0.98rem;
        line-height: 1.45;
    }}
    .csjmu-card {{
        background-color: var(--secondary-background-color);
        border-left: 5px solid var(--primary-color);
        color: var(--text-color);
        padding: 14px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
        font-size: 0.95rem;
    }}
    .csjmu-card.justify {{
        text-align: justify;
        text-justify: inter-word;
    }}
    [data-testid="stChatMessage"] {{
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {COLLEGE_SHORT} Assistant")
    st.caption(COLLEGE_NAME)

    st.markdown("#### Appearance")
    st.caption(
        "Click the **⋮** icon (top-right) → **Settings** → **Theme**, "
        "then choose **Light**, **Dark**, or **System**."
    )

    st.markdown("#### About")
    st.markdown(
        f"""
        <div class="csjmu-card">
            Answers questions about {COLLEGE_NAME} using your uploaded PDF knowledge base
            and retrieval-augmented generation (RAG).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Focus areas")
    st.markdown(
        """
        <div class="csjmu-card">
            Admissions · Academics · Calendars · Programs · Campus services
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Purpose")
    st.markdown(
        f"""
        <div class="csjmu-card justify">
            This assistant helps students and visitors find consistent, document-grounded
            information about {COLLEGE_NAME}, reducing repetitive queries and pointing users
            to the right offices when something is not in the knowledge base.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Session")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.caption(f"Messages in this session: {len(st.session_state.chat_history)}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        if "conversational_chain" in st.session_state:
            del st.session_state.conversational_chain
        st.rerun()

    st.markdown("---")
    st.markdown("#### Export")

    if st.button("Export chat to PDF", use_container_width=True):
        if len(st.session_state.chat_history) > 0:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(
                    0,
                    10,
                    text_for_pdf(f"{COLLEGE_SHORT} - Conversation export"),
                    ln=True,
                    align="C",
                )
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(
                    0,
                    8,
                    text_for_pdf(
                        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ),
                    ln=True,
                    align="C",
                )
                pdf.ln(6)

                for message in st.session_state.chat_history:
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(
                        0,
                        8,
                        text_for_pdf(message["role"].capitalize()),
                        ln=True,
                    )
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, text_for_pdf(message["content"]))
                    pdf.ln(4)

                filename = f"csjmu_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)

                with open(filename, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                os.remove(filename)
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
        else:
            st.warning("No messages to export yet.")

# Main area
st.markdown(
    f"""
    <div class="csjmu-hero">
        <h1>{COLLEGE_SHORT} · University Assistant</h1>
        <p>{COLLEGE_NAME} — ask about admissions, academics, holidays, syllabus, and more.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.image(
    "https://csjmurec.samarth.edu.in/logo.png",
    use_container_width=True,
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("### Quick Help")

# ---- INPUT HANDLING ----
user_input = st.chat_input(
    f"Ask about {COLLEGE_SHORT}…",
    key="main_chat_input"
)

# Buttons override input
col1, col2, col3 = st.columns(3)

if col1.button("Admission Process", key="btn_admission"):
    user_input = "Tell me the admission process"

if col2.button("Courses Offered", key="btn_courses"):
    user_input = "What courses are available?"

if col3.button("Fee Structure", key="btn_fees"):
    user_input = "Explain fee structure"
if user_input:
    # user message show
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # assistant response
    with st.chat_message("assistant"):
        if contains_sensitive_topics(user_input):
            assistant_response = "Restricted content"
if user_input:
    # show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response (ONLY ONCE)
    with st.chat_message("assistant"):
        if contains_sensitive_topics(user_input):
            assistant_response = "Restricted content"

            st.markdown(assistant_response)

        else:
            response = st.session_state.conversational_chain({"question": user_input})
            assistant_response = response["answer"]

            st.markdown(assistant_response)

            # show sources
            if response.get("source_documents"):
                with st.expander("Sources"):
                    for doc in response["source_documents"]:
                        st.write(doc.metadata.get("source", "Unknown"))

    # save chat
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )