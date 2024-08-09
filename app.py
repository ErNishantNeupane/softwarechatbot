import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure API key is not None or empty
if not google_api_key:
    raise ValueError("Google API key is not set. Please check your environment variable.")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    # Create embeddings instance
    embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")

    # Load the FAISS index with embeddings
    new_db = FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search and get the most relevant document
    docs = new_db.similarity_search(user_question)
    if docs:
        most_relevant_doc = docs[0]

        # Display the most relevant context
        st.write("Most Relevant Context: ")
        st.write(most_relevant_doc.page_content)

        # Generate response
        chain = get_conversational_chain()
        response = chain({"input_documents": [most_relevant_doc], "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])
    else:
        st.write("No relevant context found.")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using LLAMA 3💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
