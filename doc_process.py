import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import mysql.connector
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 

# Kết nối MySQL
def connect_mysql():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

class DocumentProcessor:
    def __init__(self, path, clear_db=False):
        self.path = path
        self.clear_db = clear_db
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.create_table()

    def create_table(self):
        conn = connect_mysql()
        cursor = conn.cursor()
        
        if self.clear_db:
            cursor.execute("DROP TABLE IF EXISTS documents")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content TEXT,
            vector LONGTEXT
        )
        """)
        conn.commit()
        conn.close()

    def load_documents(self):
        documents = []

        def load_file(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return []  

            return loader.load()

        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                file_path = os.path.join(self.path, file)
                if os.path.isfile(file_path):  
                    documents.extend(load_file(file_path))
        elif os.path.isfile(self.path):
            documents.extend(load_file(self.path))

        return documents

    def clean_text(self, text):
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def process_documents(self, documents, c_size=512, c_overlap=32, char_split='\n'):
        text_splitter = CharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separator=char_split
        )

        cleaned_docs = []
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content) 
            cleaned_docs.append(doc)

        texts = text_splitter.split_documents(cleaned_docs)

        return [doc.page_content for doc in texts]

    def save_to_mysql(self, doc_content, vector):
        conn = connect_mysql()
        cursor = conn.cursor()

        vector_json = json.dumps(vector)
        # Lưu vào MySQL
        cursor.execute("INSERT INTO documents (content, vector) VALUES (%s, %s)", (doc_content, vector_json))
        conn.commit()
        conn.close()

    def process_and_store_documents(self):
        docs = self.load_documents()
        split_docs = self.process_documents(docs)

        # Lưu mỗi document và vector vào MySQL
        for doc in split_docs:
            vector = self.embeddings.embed_query(doc)
            self.save_to_mysql(doc, vector)
        
        print("Document saved")

    def retrieve_from_db(self):
        conn = connect_mysql()
        cursor = conn.cursor()

        cursor.execute("SELECT id, content, vector FROM documents")
        rows = cursor.fetchall()

        documents = []
        for row in rows:
            doc = {
                'id': row[0],
                'content': row[1],
                'vector': json.loads(row[2])  # Chuyển chuỗi JSON về lại vector
            }
            documents.append(doc)

        conn.close()
        return documents

    def search_cosine_similarity(self, query, top_k=3):
        query_vector = self.embeddings.embed_query(query)

        documents = self.retrieve_from_db()

        vectors = [np.array(doc['vector']) for doc in documents]
        similarities = cosine_similarity([query_vector], vectors)[0]

        top_k_indices = similarities.argsort()[-top_k:][::-1]
        top_k_results = []
        for idx in top_k_indices:
            top_k_results.append(documents[idx]['content'])
        return top_k_results
