import sys
import os
import json
import logging
import fitz  # Importa o PyMuPDF (fitz)
from pathlib import Path
from typing import List, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QTextEdit, QProgressBar, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QStackedWidget, QLabel,
    QHeaderView, QSplitter, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
# QShortcut e QKeySequence são necessários para Ctrl++ / Ctrl+-
from PyQt6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
from PyQt6.QtCore import QThread, QUrl, Qt

# Importa o Worker (que executa a extração em outra thread)
from worker import Worker
# Importa seu Extractor (real ou mock)
from extractor import Extractor

# Configura um logger para este módulo
logger = logging.getLogger(__name__)


# --- Widget de Visualização de PDF Customizado ---
class PdfViewer(QGraphicsView):
    """
    Um QGraphicsView customizado que lida com zoom (Ctrl+Scroll/Pinch) 
    e pan (clicar e arrastar) para exibir um QPixmap.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        # Configurações para uma navegação suave
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Habilita "clicar e arrastar"
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # --- MUDANÇA: Fatores de Zoom (Menos Sensível) ---
        self.zoom_factor_in = 1.1  # Zoom de 10% (era 1.2)
        self.zoom_factor_out = 0.9 # Zoom de 10% (era 0.8)
        # -----------------------------------------------

    def set_pixmap(self, pixmap: QPixmap):
        """Define o pixmap e aplica o 'fit_to_window' inicial."""
        self._pixmap_item.setPixmap(pixmap)
        QApplication.processEvents() 
        self.fit_to_window()

    def fit_to_window(self):
        """Reseta o zoom e a posição para "caber na janela"."""
        if not self._pixmap_item.pixmap():
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_pixmap(self):
        """Limpa a imagem."""
        self._pixmap_item.setPixmap(QPixmap())

    # --- MUDANÇA: Métodos de Zoom Expostos ---
    def zoom_in(self):
        """Aplica o zoom in."""
        self.scale(self.zoom_factor_in, self.zoom_factor_in)

    def zoom_out(self):
        """Aplica o zoom out."""
        self.scale(self.zoom_factor_out, self.zoom_factor_out)
    # -----------------------------------------

    def wheelEvent(self, event):
        """Captura o Ctrl+Scroll (e pinch-to-zoom) para zoom."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()  # Chama o novo método
            elif delta < 0:
                self.zoom_out() # Chama o novo método
            event.accept()
        else:
            super().wheelEvent(event)
# --- Fim do Widget Customizado ---


class MainWindow(QMainWindow):

    def __init__(self, debug_mode=False):
        """
        Inicializador da Janela Principal.
        """
        super().__init__()

        # --- 1. Configuração do Estado Interno ---
        self.debug_mode = debug_mode
        self.processing_queue: List[Dict[str, Any]] = []
        self.base_dir_path: Path | None = None
        self.current_job_index: int = 0
        self.extraction_results: List[Dict[str, Any]] = []

        # (Estado de zoom foi removido daqui)

        try:
            self.extractor = Extractor()
        except Exception as e:
            logger.critical(f"Falha ao carregar o Extractor: {e}", exc_info=self.debug_mode)
            QMessageBox.critical(self, "Erro Crítico",
                                 f"Não foi possível carregar os modelos de IA. O programa será fechado.\n\nErro: {e}")
            sys.exit(1)

        self.thread: QThread | None = None
        self.worker: Worker | None = None

        # --- 2. Configuração da Janela ---
        self.setWindowTitle(f"Ferramenta de Extração de Documentos {'(MODO DEBUG)' if debug_mode else ''}")
        self.setGeometry(100, 100, 1200, 800)

        # --- 3. Inicializa e Estiliza a UI ---
        self._init_ui()
        self._update_button_states(is_busy=False)
        self._setup_shortcuts()  # <-- Configura os atalhos (Ctrl+, Ctrl-)

        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QLineEdit, QTextEdit, QTableWidget {
                background-color: #3C3C3C;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit:read-only {
                background-color: #2E2E2E;
            }
            QPushButton {
                background-color: #5C5C5C;
                border: 1px solid #777;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #6C6C6C;
            }
            QPushButton:pressed {
                background-color: #4C4C4C;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
            QProgressBar {
                text-align: center;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #4C4C4C;
                padding: 4px;
                border: 1px solid #555;
            }
            QTableWidget::item {
                padding: 3px;
            }
            QLabel {
                padding-top: 6px;
            }
            /* Estilo para o visualizador de PDF (QGraphicsView) */
            QGraphicsView {
                background-color: #202020;
                border-radius: 4px;
                border: 1px solid #444;
            }
            #schema_display_box {
                background-color: #252525;
            }
            #pdf_placeholder {
                font-size: 14pt;
                color: #888;
            }
            
            /* Estilo para o botão de reset (discreto e moderno) */
            #reset_view_button {
                background-color: rgba(92, 92, 92, 0.5); /* 50% transparente */
                border: 1px solid rgba(119, 119, 119, 0.5);
                border-radius: 15px; /* Metade do tamanho (30px) */
                font-size: 14pt;
                font-weight: bold;
                padding-bottom: 3px; /* Centraliza o ícone */
            }
            #reset_view_button:hover {
                background-color: rgba(108, 108, 108, 0.8); /* 80% opaco */
            }
            #reset_view_button:pressed {
                background-color: rgba(76, 76, 76, 0.8);
            }
            #reset_view_button:disabled {
                background-color: rgba(68, 68, 68, 0.3);
            }
        """)

    def _init_ui(self):
        """Cria e organiza todos os widgets da UI."""

        # === 1. Widgets de Carregamento (Topo) ===
        self.job_file_edit = QLineEdit()
        self.job_file_edit.setReadOnly(True)
        self.job_file_btn = QPushButton("Carregar Fila (.json)...")

        self.pdf_dir_edit = QLineEdit()
        self.pdf_dir_edit.setReadOnly(True)
        self.pdf_dir_btn = QPushButton("Alterar Diretório Base...")

        top_layout = QGridLayout()
        top_layout.addWidget(QLabel("Arquivo de Fila:"), 0, 0)
        top_layout.addWidget(self.job_file_edit, 0, 1)
        top_layout.addWidget(self.job_file_btn, 0, 2)
        top_layout.addWidget(QLabel("Diretório Base:"), 1, 0)
        top_layout.addWidget(self.pdf_dir_edit, 1, 1)
        top_layout.addWidget(self.pdf_dir_btn, 1, 2)

        # === 2. Painéis Principais (Esquerda/Direita) ===

        # Esquerda: Splitter vertical para PDF e Schema
        pdf_viewer_container = QWidget()
        
        self.pdf_viewer = PdfViewer() 
        self.pdf_viewer.setMinimumSize(400, 300)

        self.pdf_placeholder_label = QLabel("PDF será exibido aqui", pdf_viewer_container)
        self.pdf_placeholder_label.setObjectName("pdf_placeholder")
        self.pdf_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Faz o placeholder preencher o container para ser centralizado
        self.pdf_placeholder_label.setGeometry(0, 0, 400, 300) # (Será redimensionado pelo layout)
        self.pdf_placeholder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        self.reset_view_btn = QPushButton("⟲", pdf_viewer_container)
        self.reset_view_btn.setObjectName("reset_view_button")
        self.reset_view_btn.setFixedSize(30, 30)
        self.reset_view_btn.setToolTip("Resetar zoom (Ctrl+0)")
        
        overlay_layout = QGridLayout(pdf_viewer_container)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addWidget(self.pdf_viewer, 0, 0)
        overlay_layout.addWidget(self.pdf_placeholder_label, 0, 0)
        overlay_layout.addWidget(self.reset_view_btn, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        overlay_layout.setContentsMargins(0, 10, 10, 0)

        self.schema_display = QTextEdit()
        self.schema_display.setReadOnly(True)
        self.schema_display.setObjectName("schema_display_box")
        self.schema_display.setMinimumSize(400, 100)
        self.schema_display.setPlaceholderText("O schema da extração aparecerá aqui...")

        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.addWidget(pdf_viewer_container) 
        self.v_splitter.addWidget(self.schema_display)
        self.v_splitter.setSizes([400, 400])

        self.pdf_file_label = QLabel("Nenhuma fila carregada.")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.v_splitter, stretch=1)
        left_layout.addWidget(self.pdf_file_label)

        # Direita: Stack (para alternar JSON / Tabela)
        self.right_stack = QStackedWidget()
        self.json_display = QTextEdit()
        self.json_display.setReadOnly(True)
        self.right_stack.addWidget(self.json_display)
        self.progress_table = QTableWidget()
        self.progress_table.setColumnCount(3)
        self.progress_table.setHorizontalHeaderLabels(["Arquivo PDF", "Status", "Prévia do Resultado"])
        self.progress_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.progress_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.right_stack.addWidget(self.progress_table)
        mid_layout = QHBoxLayout()
        mid_layout.addLayout(left_layout, stretch=1)
        mid_layout.addWidget(self.right_stack, stretch=1)

        # === 3. Controles Inferiores (Progresso e Botões) ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.next_btn = QPushButton("Extrair Próximo")
        self.all_btn = QPushButton("Extrair Tudo")
        self.save_btn = QPushButton("Salvar Resultados...")
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.next_btn)
        bottom_layout.addWidget(self.all_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.save_btn)

        # === 4. Layout Principal (Vertical) ===
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(mid_layout, stretch=1)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(bottom_layout)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # === 5. Conectar Sinais (Cliques de Botão) ===
        self.job_file_btn.clicked.connect(self._on_load_job_file)
        self.pdf_dir_btn.clicked.connect(self._on_load_pdf_dir)
        self.next_btn.clicked.connect(self._on_extract_next)
        self.all_btn.clicked.connect(self._on_extract_all)
        self.save_btn.clicked.connect(self._on_save_results)
        self.reset_view_btn.clicked.connect(self._on_view_reset)

    # --- Funções de Atalho e Eventos ---
    
    def _setup_shortcuts(self):
        """Configura os atalhos de teclado para zoom."""
        # Conecta os atalhos da janela principal aos métodos de zoom
        # do nosso PdfViewer customizado.
        QShortcut(QKeySequence("Ctrl++"), self).activated.connect(self.pdf_viewer.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(self.pdf_viewer.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self.pdf_viewer.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(self._on_view_reset)

    # (wheelEvent foi movido para a classe PdfViewer)

    # --- Funções de Lógica (Slots) ---

    def _on_load_job_file(self):
        """Abre um diálogo para carregar o arquivo JSON da fila."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Fila de Extração", "", "JSON Files (*.json)")
        if file_path:
            job_file_path = Path(file_path)
            try:
                with open(job_file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                if not (isinstance(loaded_data, list) and loaded_data and
                        isinstance(loaded_data[0], dict) and
                        'label' in loaded_data[0] and
                        'extraction_schema' in loaded_data[0] and
                        'pdf_path' in loaded_data[0]):
                    raise ValueError(
                        "A estrutura do JSON está incorreta. Faltam chaves: 'label', 'extraction_schema', ou 'pdf_path'.")

                self.processing_queue = loaded_data
                self.current_job_index = 0
                self.extraction_results.clear()
                self.json_display.clear()
                self.schema_display.clear()
                self.progress_bar.setValue(0)
                self.job_file_edit.setText(str(job_file_path))

                self.base_dir_path = job_file_path.parent
                self.pdf_dir_edit.setText(str(self.base_dir_path))

                self.pdf_viewer.clear_pixmap()
                self.pdf_placeholder_label.setText("PDF será exibido aqui")
                self.pdf_placeholder_label.show() # Mostra o placeholder

                QMessageBox.information(self, "Fila Carregada",
                                        f"{len(self.processing_queue)} trabalhos carregados.\n\nDiretório Base definido como:\n{self.base_dir_path}")

            except Exception as e:
                logger.error(f"Falha ao carregar o arquivo de fila: {e}", exc_info=self.debug_mode)
                QMessageBox.critical(self, "Erro", f"Falha ao carregar ou validar o arquivo de fila: {e}")
                self.processing_queue = []
                self.job_file_edit.setText("")
                self.pdf_dir_edit.setText("")

            self._update_button_states()

    def _on_load_pdf_dir(self):
        """Permite ao usuário sobrescrever o Diretório Base."""
        start_dir = str(self.base_dir_path) if self.base_dir_path else ""
        dir_path = QFileDialog.getExistingDirectory(self, "Selecionar Diretório Base de PDFs", start_dir)

        if dir_path:
            self.base_dir_path = Path(dir_path)
            self.pdf_dir_edit.setText(str(self.base_dir_path))
            logger.info(f"Diretório Base sobrescrito pelo usuário para: {self.base_dir_path}")
            self._update_button_states()

    def _update_button_states(self, is_busy=False):
        """Ativa/Desativa botões baseado no estado do app."""
        self.job_file_btn.setEnabled(not is_busy)

        has_jobs = bool(self.processing_queue)
        has_base_dir = self.base_dir_path is not None
        has_jobs_remaining = self.current_job_index < len(self.processing_queue)
        has_results = bool(self.extraction_results)

        self.pdf_dir_btn.setEnabled(not is_busy and has_jobs)
        self.next_btn.setEnabled(not is_busy and has_jobs and has_base_dir and has_jobs_remaining)
        self.all_btn.setEnabled(not is_busy and has_jobs and has_base_dir and has_jobs_remaining)
        self.save_btn.setEnabled(not is_busy and has_results)
        
        has_image = self.pdf_viewer._pixmap_item.pixmap() is not None
        self.reset_view_btn.setEnabled(has_image and not is_busy)

    def _display_pdf_as_image(self, pdf_path: Path):
        """Renderiza a pág 1 e passa o QPixmap para o PdfViewer."""
        try:
            with fitz.open(pdf_path) as fitz_doc:
                page = fitz_doc[0]
                pix = page.get_pixmap(dpi=150)

                image_format = QImage.Format.Format_RGB888
                if pix.alpha:
                    image_format = QImage.Format.Format_RGBA8888

                q_image = QImage(pix.samples, pix.width, pix.height, pix.stride, image_format)
                full_res_pixmap = QPixmap.fromImage(q_image)

                self.pdf_placeholder_label.hide() # Esconde o placeholder
                self.pdf_viewer.set_pixmap(full_res_pixmap) # Mostra a imagem

        except Exception as e:
            logger.error(f"Falha ao renderizar PDF {pdf_path.name}: {e}", exc_info=self.debug_mode)
            QMessageBox.critical(self, "Erro ao Renderizar PDF", f"Não foi possível exibir o PDF: {e}")
            self.pdf_viewer.clear_pixmap()
            self.pdf_placeholder_label.setText(f"Falha ao carregar:\n{pdf_path.name}")
            self.pdf_placeholder_label.show() # Mostra o placeholder de erro
        
        self._update_button_states()

    def _on_view_reset(self):
        """Slot para o botão de reset (chama o método do viewer)."""
        logger.debug("Resetando a visualização do PDF")
        self.pdf_viewer.fit_to_window()

    def _on_extract_next(self):
        """Inicia a extração de um ÚNICO trabalho."""
        job = self.processing_queue[self.current_job_index]
        try:
            pdf_path = self.base_dir_path / job['pdf_path']
            if not pdf_path.exists():
                QMessageBox.warning(self, "Arquivo Não Encontrado",
                                    f"Não foi possível encontrar o PDF: {pdf_path}\n\nVerifique seu Diretório Base.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Erro de Caminho", f"Erro ao construir o caminho do PDF: {e}")
            return

        self._display_pdf_as_image(pdf_path)

        self.pdf_file_label.setText(
            f"Trabalho {self.current_job_index + 1} de {len(self.processing_queue)}: {pdf_path.name}")

        schema = job['extraction_schema']
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        self.schema_display.setText(schema_str)

        self.right_stack.setCurrentWidget(self.json_display)
        self.json_display.append(f"\n--- Extraindo {pdf_path.name}... ---")

        self.progress_bar.setRange(0, 0)
        self._start_worker_thread([job])

    def _on_extract_all(self):
        """Inicia a extração de TODOS os trabalhos restantes."""
        jobs_to_process = self.processing_queue[self.current_job_index:]
        if not jobs_to_process:
            return

        self._setup_progress_table(jobs_to_process)
        self.right_stack.setCurrentWidget(self.progress_table)

        self.progress_bar.setRange(0, len(jobs_to_process))
        self.progress_bar.setValue(0)

        self.pdf_viewer.clear_pixmap()
        self.pdf_placeholder_label.setText("Processando em lote...")
        self.pdf_placeholder_label.show() # Mostra o placeholder
        
        self.schema_display.setText("Exibição de schema pausada durante o processamento em lote.")

        self.pdf_file_label.setText(f"Processando {len(jobs_to_process)} trabalhos...")
        
        self._update_button_states(is_busy=True)
        self._start_worker_thread(jobs_to_process)

    def _setup_progress_table(self, jobs: list):
        """Preenche a tabela para o modo 'Extrair Tudo'."""
        self.progress_table.clearContents()
        self.progress_table.setRowCount(len(jobs))
        for i, job in enumerate(jobs):
            relative_path = job['pdf_path']
            self.progress_table.setItem(i, 0, QTableWidgetItem(relative_path))
            self.progress_table.setItem(i, 1, QTableWidgetItem("Pendente..."))
            self.progress_table.setItem(i, 2, QTableWidgetItem(""))

    def _start_worker_thread(self, jobs_to_process: list):
        """Cria e inicia a QThread e o Worker."""
        self._update_button_states(is_busy=True)

        self.thread = QThread()
        self.worker = Worker(
            extractor=self.extractor,
            base_dir=self.base_dir_path
        )
        self.worker.set_jobs(jobs_to_process)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_extraction_finished)
        self.worker.result_ready.connect(self._on_single_result)
        self.worker.result_ready.connect(self._on_batch_result)
        self.worker.batch_progress.connect(self._on_batch_progress)
        self.worker.error.connect(self._on_error)

        self.thread.start()

    def _on_single_result(self, pdf_name: str, result_dict: dict):
        """Slot para quando UM resultado fica pronto (modo 'Próximo')."""
        if self.right_stack.currentWidget() == self.json_display:
            current_text = self.json_display.toPlainText()
            current_text = current_text.rsplit(f"\n--- Extraindo {pdf_name}... ---", 1)[0]
            self.json_display.setText(current_text.strip())

            self.json_display.append(f"\n--- {pdf_name} ---")
            self.json_display.append(json.dumps(result_dict, indent=2, ensure_ascii=False))
            self.json_display.append("-" * (len(pdf_name) + 6))

            job = self.processing_queue[self.current_job_index]
            self.extraction_results.append({
                "pdf_path_relativo": job['pdf_path'],
                "resultado": result_dict
            })
            self.current_job_index += 1

    def _on_batch_result(self, pdf_name: str, result_dict: dict):
        """Slot para quando UM resultado fica pronto (modo 'Tudo')."""
        if self.right_stack.currentWidget() == self.progress_table:
            found_row = -1
            for i in range(self.progress_table.rowCount()):
                if self.progress_table.item(i, 0).text().endswith(pdf_name):
                    self.progress_table.setItem(i, 1, QTableWidgetItem("✅ Concluído"))
                    preview = json.dumps(result_dict, ensure_ascii=False)
                    if len(preview) > 100:
                        preview = preview[:100] + "..."
                    self.progress_table.setItem(i, 2, QTableWidgetItem(preview))
                    found_row = i
                    break
            
            if found_row != -1:
                job_index_in_queue = self.current_job_index + found_row
                job = self.processing_queue[job_index_in_queue]

                self.extraction_results.append({
                    "pdf_path_relativo": job['pdf_path'],
                    "resultado": result_dict
                })
    
    def _on_batch_progress(self, current_val: int, total_val: int):
        """Atualiza a barra de progresso principal."""
        self.progress_bar.setValue(current_val)

    def _on_extraction_finished(self):
        """Limpa a thread e reativa a UI."""
        logger.info("Trabalho da thread concluído.")

        if self.right_stack.currentWidget() == self.progress_table:
            self.current_job_index = len(self.processing_queue)

        if self.thread:
            self.thread.quit()
            self.thread.wait()

        self.thread = None
        self.worker = None

        self.progress_bar.setRange(0, 100)
        if self.current_job_index == len(self.processing_queue):
            self.progress_bar.setValue(100)

        self._update_button_states(is_busy=False)

    def _on_error(self, simple_error_msg: str):
        """Mostra apenas a mensagem de erro simples no popup."""
        logger.debug(f"Slot _on_error recebendo: {simple_error_msg}")
        QMessageBox.critical(self, "Erro na Extração", simple_error_msg)

    def _on_save_results(self):
        """Salva a LISTA 'self.extraction_results' em um JSON."""
        if not self.extraction_results:
            QMessageBox.warning(self, "Nada para Salvar", "Nenhum resultado de extração foi gerado ainda.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Resultados", "", "JSON Files (*.json)")

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.extraction_results, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "Sucesso", f"Resultados salvos com sucesso em:\n{file_path}")
            except Exception as e:
                logger.error(f"Falha ao salvar o arquivo: {e}", exc_info=self.debug_mode)
                QMessageBox.critical(self, "Erro ao Salvar", f"Não foi possível salvar o arquivo: {e}")

    def closeEvent(self, event):
        """Garante que a thread pare ao fechar a janela."""
        if self.thread and self.thread.isRunning():
            logger.info("Fechando janela, parando thread de trabalho...")
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        event.accept()
