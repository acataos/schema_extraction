# worker.py
import fitz # PyMuPDF
from PyQt6.QtCore import QObject, pyqtSignal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Worker(QObject):
    """
    Um 'trabalhador' QObject que executa tarefas em uma thread separada
    para não congelar a interface.
    """
    
    # Sinais que o trabalhador pode emitir:
    
    # Sinal emitido quando um *único* arquivo é concluído
    # (nome_arquivo, dicionario_json)
    result_ready = pyqtSignal(str, dict)
    
    # Sinal emitido durante "Extrair Tudo"
    # (numero_atual, total_arquivos)
    batch_progress = pyqtSignal(int, int)
    
    # Sinal emitido quando toda a fila de trabalho termina
    finished = pyqtSignal()
    
    # Sinal emitido se ocorrer um erro
    error = pyqtSignal(str)

    def __init__(self, extractor, base_dir: Path):
        super().__init__()
        self.extractor = extractor
        self.base_dir = base_dir
        self._is_running = True
        self.jobs = []

    def set_jobs(self, jobs: list):
        """Define a lista de arquivos PDF (Path) a processar."""
        self.jobs = jobs

    def run(self):
        """O método principal que será executado na nova thread."""
        try:
            total_files = len(self.jobs)
            
            for i, job in enumerate(self.jobs):
                if not self._is_running:
                    break

                # 1. Abrir o PDF e extrair o 'dict' (trabalho do PyMuPDF)
                try:
                    label = job["label"]
                    schema = job["extraction_schema"]
                    relative_pdf_path = job["pdf_path"]
                    pdf_path = self.base_dir / relative_pdf_path
                    if not pdf_path.exists():
                        raise Exception(f"Arquivo PDF não encontrado: {pdf_path}")
                    pdf_name = pdf_path.name
                except KeyError as e:
                    raise Exception(f"Trabalho {i+1} está mal formatado. Chave ausente: {e}")
                except Exception as e:
                     raise Exception(f"Trabalho {i+1} tem um pdf_path inválido: {pdf_path_str}. Erro: {e}")

                # 2. Abre o PDF e extrai o 'dict'
                try:
                    with fitz.open(pdf_path) as fitz_doc:
                        page = fitz_doc[0] # Pega a primeira página
                        fitz_page_dict = page.get_text("dict")
                except Exception as e:
                    raise Exception(f"Falha ao ler PDF {pdf_name}: {e}")
                
                # 3. Chama 'extract' com o label/schema do trabalho
                result_dict = self.extractor.extract(
                    label, 
                    schema, 
                    fitz_page_dict
                )
                
                # 4. Emite os resultados
                self.result_ready.emit(pdf_name, result_dict)
                
                if total_files > 1:
                    self.batch_progress.emit(i + 1, total_files)

        except Exception as e:
            logger.error(f"Falha na extração: {e}")
            logger.debug(f"Stack trace completo da falha:", exc_info=True)
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False
