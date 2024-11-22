import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template  # Import your HTML templates

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger chunks to ensure more context is captured
        chunk_overlap=300,  # Increase overlap for better context continuity
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Custom Prompt for ConversationalRetrievalChain
CUSTOM_PROMPT = """
You are an assistant that answers questions using only the provided context.
If the answer is not in the context, say "I don't know."

### Context:
{context}

### Question:
{question}

Helpful Answer:
"""

# Function to get the conversation chain
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/gemma-2b-it", model_kwargs={"temperature": 0.2, "max_length": 512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # ConversationalRetrievalChain with a custom prompt
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),  # Increase to 15 documents for better context
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    try:
        # Render the user's message first
        user_message = user_template.replace("{{MSG}}", user_question)
        st.markdown(user_message, unsafe_allow_html=True)

        # Get the bot's response
        response = st.session_state.conversation({"question": user_question})

        # Render the bot's response
        bot_message = bot_template.replace("{{MSG}}", response['answer'])
        st.markdown(bot_message, unsafe_allow_html=True)

        # Debugging: Display retrieved context to ensure completeness
        with st.expander("Retrieved Context (Debug)"):
            for doc in response.get("source_documents", []):
                st.write(f"{doc.page_content[:500]}...")  # Check for truncation or missing details

        # Additional Debugging to check if the relevant context is present
        if 'Satoshi Nakamoto' not in [doc.page_content for doc in response.get("source_documents", [])]:
            st.warning("The context does not seem to contain the relevant information about Bitcoin's creator!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main application function
def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot", page_icon=":books:")

    # Inject CSS into the page to style the chat
    st.markdown(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("QueryVerse :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversational retrieval chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

# Run the application
if __name__ == '__main__':
    main()
