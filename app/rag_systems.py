import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class RAGSystem:
    def __init__(self):
        self.load_documents()
        self.create_embeddings()
        self.setup_qa_chain()

    def load_documents(self):
        loader = DirectoryLoader("data/", glob="**/*.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.texts = text_splitter.split_documents(documents)

    def create_embeddings(self):
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(self.texts, embeddings)

    def setup_qa_chain(self):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=150)
        
        template = """You are an expert in Sound, Sound waves, and Ultrasound, especially as covered in NCERT textbooks. 
        Use the following pieces of context to answer the question at the end. 
        If the information is not in the context, say "I don't have that specific information from the NCERT textbooks."
        Provide detailed, accurate answers related to Sound topics.
        
        Context:
        {context}
        
        Question: {question}
        
        Detailed answer (up to 150 tokens):"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def query(self, question: str) -> str:
        response = self.qa_chain.run(question)
        return response