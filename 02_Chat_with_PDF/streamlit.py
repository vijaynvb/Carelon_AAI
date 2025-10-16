import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from app import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
)

@st.cache_resource
def build_conversation(raw_text: str):
    # Build and cache the chain; LangChain memory persists inside this cached instance.
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    return get_conversation_chain(vectorstore)

# UI: render conversation turns
def handle_userinput(user_question: str, conversation):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                build_conversation(raw_text)  # warm cache
                st.success("Documents processed successfully. You can start asking questions.")

    # Handle questions by retrieving the cached conversation (no session_state)
    if user_question and pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        conversation = build_conversation(raw_text)
        handle_userinput(user_question, conversation)

if __name__ == '__main__':
    main()
