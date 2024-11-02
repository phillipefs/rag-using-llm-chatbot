import os
import shutil
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

from cachetools import TTLCache, cached
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import streamlit as st

class OpenAIGPT:
    """Class for interacting with OpenAI's GPT models."""

    cache = TTLCache(maxsize=100, ttl=3599)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scope: str,
        model_txt: str,
        model_embedding: str,
        token_endpoint: str,
        proxy: str,
    ):
        """
        Initialize OpenAIGPT class with required parameters.

        Args:
            client_id (str): Client ID.
            client_secret (str): Client secret.
            scope (str): Scope.
            model_txt (str): Text model name.
            model_embedding (str): Embedding model name.
            token_endpoint (str): Token endpoint URL.
            proxy (str): Proxy URL.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.model_txt = model_txt
        self.model_embedding = model_embedding
        self.token_endpoint = token_endpoint
        self.proxy = proxy
        self.db = None
        self.embedding_model = None
        self.text_model = None
        self.faiss_db_path = "faiss_db"

    @cached(cache)
    def _get_token_openai(self) -> str:
        """
        Retrieve the OpenAI API access token.
        """
        logging.info("Obtaining API Token")
        try:
            response = requests.post(
                self.token_endpoint,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": self.scope,
                    "grant_type": "client_credentials",
                },
                proxies={"http": self.proxy, "https": self.proxy},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            logging.error("Failed to obtain access token", exc_info=True)
            raise RuntimeError(f"Failed to obtain access token: {str(e)}") from e

    def _llm_text(self) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance for text generation.
        """
        logging.info("Creating Text LLM")
        if self.text_model is None:
            self.text_model = ChatOpenAI(
                api_key=self._get_token_openai(),
                base_url="https://api.pd01i.gcp.ford.com/fordllmapi/api/v1",
                model=self.model_txt,
            )
        return self.text_model

    def _llm_embedding(self) -> OpenAIEmbeddings:
        """
        Create an OpenAIEmbeddings instance for embeddings.
        """
        logging.info("Creating Embeddings LLM")
        if self.embedding_model is None:
            self.embedding_model = OpenAIEmbeddings(
                api_key=self._get_token_openai(),
                base_url="https://api.pd01i.gcp.ford.com/fordllmapi/api/v1",
                model=self.model_embedding,
            )
        return self.embedding_model

    def create_vector_from_disk(self):
        """Load vector database from disk using FAISS."""
        print("Loading Database....")
        embedding_model = self._llm_embedding()
        self.db = FAISS.load_local(self.faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
        return self.db 

    def create_vector_from_documents(self, all_documents_txt: str, batch_size: int = 600) -> FAISS:
        """
        Create a vector database from documents, handling batching to avoid overflow.
        """
        print("Creating Database....")
        
        # Generate documents from received text
        docs = [Document(page_content=all_documents_txt)]
        
        # Split documents into smaller chunks using text_splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=500)
        split_docs = text_splitter.split_documents(docs)

        print(f"TotalSplit: {len(split_docs)}")
        
        # Split documents into smaller batches to avoid buffer overflow
        batches = [split_docs[i:i + batch_size] for i in range(0, len(split_docs), batch_size)]

        print(f"Total batches: {len(batches)}")
        
        embedding_model = self._llm_embedding()
        
        # Internal function to process each batch individually
        def process_batch(batch_texts):
            return FAISS.from_texts(batch_texts, embedding_model)
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            faiss_dbs = list(executor.map(process_batch, [[doc.page_content for doc in batch] for batch in batches]))

        # Combine results from faiss_dbs
        if faiss_dbs:
            self.db = faiss_dbs[0]
            for db in faiss_dbs[1:]:
                self.db.merge_from(db)

        # # Remove o diretório existente para sobrescrever o banco de dados
        # if os.path.exists(self.faiss_db_path):
        #     shutil.rmtree(self.faiss_db_path)

        # # Cria o diretório e salva o banco de dados FAISS em disco
        # os.makedirs(self.faiss_db_path)
        # self.db.save_local(self.faiss_db_path)
                

        print("************PROCESS COMPLETED************")
        
        return self.db


    def get_response_from_documents(self, question: str) -> str:
        """
        Get a response from documents based on a given question.
        """
        llm = self._llm_text()

        # Use FAISS database from session_state
        db_vector = st.session_state['faiss_db']

        docs_similarity = db_vector.similarity_search(question, k=4)
        docs_page_content = " ".join(
            [document.page_content for document in docs_similarity]
        )

        print(docs_page_content)

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Sua especialidade é consultar documentação e responder perguntas sobre ela.",
                ),
                (
                    "human",
                    (
                        "Responda a seguinte pergunta:\n"
                        "{question}\n"
                        "Usando a seguinte fonte de dados:\n"
                        "{docs_page_content}\n\n"
                        "Use apenas a fonte de dados para responder a esta pergunta. "
                        "Se você não souber a resposta, responda apenas com: \n "
                        "A informação solicitada não está disponível nos dados recuperados. Por favor, tente outra consulta ou tópico."
                        "Suas respostas devem ser detalhadas e completas."
                        "O texto enviado é resultado de uma busca semântica, verifique bem a pergunta para não retornar informação incorreta"
                    ),
                ),
            ]
        )

        chain = chat_prompt | llm
        response = chain.invoke(
            {"question": question, "docs_page_content": docs_page_content}
        )
        return response.content
