import streamlit as st
from rag_pipeline import get_vectorstore, ask_question

st.set_page_config(
    page_title="TechDocs AI",
    page_icon="💻",
    layout="wide"
)

st.title("💻 TechDocs AI")
st.caption("Python Documentation RAG Assistant")


# Sidebar information
st.sidebar.header("About TechDocs AI")
st.sidebar.write(
    "Ask questions about Python using official documentation."
)
st.sidebar.write(
    "Powered by LangChain, FAISS, Hugging Face and Gemini."
)

# Initialize RAG vector database
@st.cache_resource
def initialize():
    return get_vectorstore()


try:
    with st.spinner("Loading Python documentation..."):
        vectorstore = initialize()
except Exception as e:
    st.error(f"Document indexing failed: {e}")
    st.stop()


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Ask a Python question...")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            
            try:
                answer, sources = ask_question(
                    vectorstore,
                    question
                )

                # Clean Gemini response
                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                    )
                elif not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

                if sources:
                    st.caption(
                        "Retrieved sources: " + ", ".join(sources)
                    )

                saved_answer = answer

            except Exception as e:
                saved_answer = f"Error: {e}"
                st.error(saved_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": saved_answer
            })

