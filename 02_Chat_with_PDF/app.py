from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize environment
load_dotenv()

# Backend: extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Backend: split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Backend: build FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Backend: create conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = ChatBedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        model_kwargs={"temperature": 0.5}
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
