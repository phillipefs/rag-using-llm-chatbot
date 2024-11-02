import os
import PyPDF2
from concurrent.futures import ThreadPoolExecutor

class PDFReader:
    """
    A class to read and concatenate text from PDF files in a given directory.
    """

    def __init__(self):
        """
        Initializes the PDFReader instance.
        """
        pass

    def _extract_text_from_pdf(self, path_file: str) -> str:
        """
        Extracts text from a single PDF file.
        """
        try:
            with open(path_file, 'rb') as file_pdf:
                reader_pdf = PyPDF2.PdfReader(file_pdf)
                text = ''
                for page in reader_pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            print(f"Erro ao processar {path_file}: {e}")
            return ''

    def concatenate_documents(self, directory: str, max_workers: int = 12) -> str:
        """
        Concatenates text from all PDF files in the specified directory using multithreading.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' not found.")

        all_documents_txt = ''

        # Lists PDFs in the directory
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

        # Processes PDFs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self._extract_text_from_pdf, pdf_files)

        # Concatenates text from all PDFs
        for result in results:
            all_documents_txt += result + '\n\n'

        return all_documents_txt