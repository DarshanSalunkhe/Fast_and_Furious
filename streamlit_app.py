import streamlit as st
import requests

import ast
import json


# ============================================================
# CONFIGURATION
# ============================================================

RAG_API_URL = "https://fast-and-furious-fl85.onrender.com/ask"

st.set_page_config(
    page_title="Fast & Furious RAG",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #0b0b0f;
        color: #ffffff;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 7rem;
    }

    /* Hide Streamlit default elements */

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .car-logo {
        text-align: center;
        font-size: 58px;
        line-height: 1;
        margin-bottom: 12px;
    }

    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #b8b8c2;
        font-size: 21px;
        font-weight: 500;
        margin-bottom: 12px;
    }

    .description {
        text-align: center;
        color: #8e8e99;
        font-size: 16px;
        line-height: 1.5;
        margin: 0 auto 35px auto;
        max-width: 700px;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 12px;
    }

    /* User message */

    div[data-testid="stChatMessage"]:has(
        div[data-testid="chatAvatarIcon-user"]
    ) {
        background: #25252d;
    }

    /* Assistant message */

    div[data-testid="stChatMessage"]:has(
        div[data-testid="chatAvatarIcon-assistant"]
    ) {
        background: #15151c;
        border: 1px solid #292932;
    }


    /* Message text */

    div[data-testid="stChatMessage"] p {
        color: #f5f5f7 !important;
        font-size: 16px;
        line-height: 1.6;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    div[data-testid="stChatInput"] {
        background: #181820 !important;
        border: 1px solid #34343e !important;
        border-radius: 16px !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background: #181820 !important;
        caret-color: #ffffff !important;
        font-size: 16px !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8e8e99 !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        color: #ffffff !important;
    }


    /* ========================================================
       INPUT SEND BUTTON
       ======================================================== */

    div[data-testid="stChatInput"] button {
        color: #ffffff !important;
    }


    /* ========================================================
       SPINNER
       ======================================================== */

    div[data-testid="stSpinner"] {
        color: #ffffff !important;
    }


    /* ========================================================
       ERROR BOX
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ========================================================
       SOURCE EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {
        background: #121219;
        border: 1px solid #292932;
        border-radius: 12px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 600px) {

        .block-container {
            padding-top: 1.8rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 34px;
        }

        .subtitle {
            font-size: 18px;
        }

        .description {
            font-size: 14px;
        }

        .car-logo {
            font-size: 48px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="car-logo">🏎️</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Fast & Furious</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">RAG Knowledge Database</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="description">
        Ask me anything about the Fast & Furious franchise,
        characters, movies, cars, and events!
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================



def clean_response(raw_answer):
    """
    Extract only the final text from Gemini's response.
    Handles:
    1. Actual Python lists
    2. JSON strings
    3. Python-representation strings
    4. Normal text
    """

    # --------------------------------------------------------
    # CASE 1: Already a list
    # --------------------------------------------------------

    if isinstance(raw_answer, list):

        text_parts = []

        for block in raw_answer:

            if not isinstance(block, dict):
                continue

            # Ignore thinking/reasoning
            if block.get("type") == "thinking":
                continue

            # Keep final text
            if block.get("type") == "text":
                text = block.get("text", "")

                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()


    # --------------------------------------------------------
    # CASE 2: String
    # --------------------------------------------------------

    if isinstance(raw_answer, str):

        cleaned = raw_answer.strip()

        # Try JSON first
        try:
            parsed = json.loads(cleaned)

            if isinstance(parsed, list):
                return clean_response(parsed)

        except Exception:
            pass


        # Try Python list representation
        try:
            parsed = ast.literal_eval(cleaned)

            if isinstance(parsed, list):
                return clean_response(parsed)

        except Exception:
            pass


        # Normal text
        return cleaned


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return str(raw_answer)# ============================================================
# CALL RAG API
# ============================================================

def ask_rag(question):
    """
    Sends the user's question to the Fast & Furious
    FastAPI RAG backend.
    """

    try:

        response = requests.post(
            RAG_API_URL,
            json={
                "question": question
            },
            timeout=120,
        )

        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        if response.status_code == 200:

            data = response.json()

            raw_answer = data.get(
                "answer",
                "No answer was returned by the RAG system.",
            )

            answer = clean_response(raw_answer)

            status = data.get(
                "status",
                "unknown",
            )

            sources = data.get(
                "sources",
                [],
            )

            return {
                "success": True,
                "answer": answer,
                "status": status,
                "sources": sources,
            }


        # ----------------------------------------------------
        # API error
        # ----------------------------------------------------

        return {
            "success": False,
            "answer": (
                f"The RAG server returned an error "
                f"(HTTP {response.status_code})."
            ),
            "status": "error",
            "sources": [],
        }


    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        return {
            "success": False,
            "answer": (
                "The RAG server took too long to respond. "
                "Please try again."
            ),
            "status": "timeout",
            "sources": [],
        }


    # --------------------------------------------------------
    # Connection error
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "answer": (
                "I couldn't connect to the Fast & Furious "
                "RAG server. The backend may be starting up."
            ),
            "status": "connection_error",
            "sources": [],
        }


    # --------------------------------------------------------
    # Other request errors
    # --------------------------------------------------------

    except requests.exceptions.RequestException as error:

        return {
            "success": False,
            "answer": "Unable to connect to the RAG backend.",
            "status": "request_error",
            "sources": [],
        }


    # --------------------------------------------------------
    # Unexpected error
    # --------------------------------------------------------

    except Exception as error:

        return {
            "success": False,
            "answer": "An unexpected error occurred.",
            "status": "error",
            "sources": [],
        }


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

        # Display sources only for assistant messages
        if role == "assistant":

            sources = message.get("sources", [])

            if sources:

                unique_sources = list(
                    dict.fromkeys(sources)
                )

                with st.expander("📚 Retrieved Sources"):

                    for source in unique_sources:
                        st.write(f"• {source}")


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "E.g., Who is Dominic Toretto?"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        # ----------------------------------------------------
        # Store and display user question
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        # ----------------------------------------------------
        # Get RAG response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching the Fast & Furious knowledge base..."
            ):

                result = ask_rag(question)


            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            if result["success"]:

                st.markdown(
                    result["answer"]
                )

            else:

                st.error(
                    result["answer"]
                )


            # ------------------------------------------------
            # Display sources
            # ------------------------------------------------

            sources = result.get(
                "sources",
                []
            )

            if sources:

                unique_sources = list(
                    dict.fromkeys(sources)
                )

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for source in unique_sources:

                        st.write(
                            f"• {source}"
                        )


        # ----------------------------------------------------
        # Store assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get(
                    "sources",
                    []
                ),
            }
        )
