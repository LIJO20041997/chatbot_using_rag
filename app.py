import streamlit as st
import time
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

st.title("CHATBOT FOR WEBSITES USING RAG")

# Allow the user to input a single URL
url_input = st.text_input("Enter a URL")

# Allow the user to input a question
question_input = st.text_input("Ask me anything:")

# Button to submit and generate the answer
if st.button("Submit"):

    if url_input and question_input:
        # Load the document from the URL
        loader = UnstructuredURLLoader(urls=[url_input])
        data = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        all_splits = docs

        # Create vectorstore from documents
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),
            persist_directory="DB"
        )

        # Set up retriever for document retrieval
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # Set up the language model for generating responses
        llm = OpenAI(temperature=0.4, max_tokens=500)

        # Define the system prompt for the assistant
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create retrieval chain for processing the query
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Get the answer from the chain
        response = rag_chain.invoke({"input": question_input})

        # Display the answer
        st.write(response["answer"])
    else:
        st.write("Please enter both a URL and a question.")
