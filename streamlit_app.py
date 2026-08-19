import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fast & Furious RAG",
    page_icon="🏎️",
    layout="centered"
)


# ============================================================
# RAG BACKEND URL
# ============================================================

RAG_API_URL = "https://fast-and-furious-fl85.onrender.com/ask"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0b0f;
        color: white;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 850px;
    }

    .logo {
        text-align: center;
        font-size: 65px;
        margin-bottom: 5px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #b9b9c3;
    }

    .description {
        text-align: center;
        font-size: 16px;
        color: #8f8f9b;
        margin-bottom: 35px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="logo">🏎️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Fast & Furious</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">RAG Knowledge Database</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description">
        Ask me anything about the Fast & Furious franchise,
        characters, movies, cars, and events!
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "E.g., Who is Dominic Toretto?"
)


# ============================================================
# SEND QUESTION TO RAG API
# ============================================================

if question:

    # Show user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # Ask FastAPI RAG backend
    with st.chat_message("assistant"):

        with st.spinner("Searching the Fast & Furious knowledge base..."):

            try:

                response = requests.post(
                    RAG_API_URL,
                    json={
                        "question": question
                    },
                    timeout=120
                )


                # Check HTTP response
                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer was returned."
                    )

                    status = data.get(
                        "status",
                        "unknown"
                    )


                    # Display answer
                    st.markdown(answer)


                    # Optional source information
                    sources = data.get("sources", [])

                    if sources:
                        with st.expander("Retrieved Sources"):
                            for source in sources:
                                st.write(f"• {source}")


                else:

                    answer = (
                        f"RAG API returned an error "
                        f"(HTTP {response.status_code})."
                    )

                    st.error(answer)


            except requests.exceptions.Timeout:

                answer = (
                    "The RAG server took too long to respond. "
                    "Please try again."
                )

                st.error(answer)


            except requests.exceptions.RequestException as e:

                answer = (
                    "Unable to connect to the Fast & Furious "
                    "RAG backend."
                )

                st.error(answer)

                st.caption(str(e))


            except Exception as e:

                answer = "An unexpected error occurred."

                st.error(answer)

                st.caption(str(e))


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
