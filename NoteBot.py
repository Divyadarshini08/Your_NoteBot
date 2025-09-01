```py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

OpenAPIKey=""

st.header("Your NoteBot")

with st.sidebar:
    st.title("Your Notes")
    file=st.file_uploader("Upload Notes PDF",type="pdf")

if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text+=page.extract_text()

    splitter=RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=200,chunk_overlap=50)
    chunks=splitter.split_text(text)
    st.write(chunks)

    embeddings=OpenAIEmbeddings(api_key=OpenAPIKey)

    vector_store=FAISS.from_texts(chunks,embeddings)

    #get user query

    user_query=st.text_input("Type your Query")

    if user_query:
        matching_chunks=vector_store.similarity_search(user_query) #similarity_search converts user query into embedding


        #llm defining
        llm=ChatOpenAI(
            api_key=OpenAPIKey,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        chain=load_qa_chain(llm,chain_type="stuff")
        output=chain.run(question=user_query,documents=matching_chunks)
        st.write(output)




