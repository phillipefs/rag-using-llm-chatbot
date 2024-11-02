import os
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader  # Atualize para o novo local do loader

class PDFReader:
    """
    A class to read and concatenate text from PDF files in a given directory.
    """
    def __init__(self):
        """
        Initializes the PDFReader instance.
        """
        pass

    def _load_pdf(self, file):
        """
        Helper method to load a single PDF file.
        """
        doc = PyPDFLoader(file)
        return doc.load()

    def return_list_pdf_documents(self, directory) -> list:
        """
        Reads and concatenates text from all PDF files in the specified directory.
        """
        docs = []
        files_path = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(self._load_pdf, files_path)
            for result in results:
                docs.extend(result)

        return docs

# Adicione o bloco principal
if __name__ == "__main__":
    start_time = time.time()  # Marca o início do tempo
    
    pdf_reader = PDFReader()
    pdf_docs = pdf_reader.return_list_pdf_documents('documents_full')
    
    end_time = time.time()  # Marca o fim do tempo
    elapsed_time = end_time - start_time  # Calcula o tempo total
    
    print(f"Tempo total para carregar documentos PDF: {elapsed_time:.2f} segundos")
