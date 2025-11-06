# app.py
import sys
import logging
import argparse
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

def setup_logging(debug=False):
    """Configura o nível de logging global."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] {%(levelname)s} (%(threadName)s) %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.info(f"Nível de logging definido para: {'DEBUG' if debug else 'INFO'}")

if __name__ == "__main__":
    # 1. Configura o parser de argumentos
    parser = argparse.ArgumentParser(description="Ferramenta de Extração de Documentos")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Ativa o logging de depuração (imprime stack traces de erros no console)"
    )
    args = parser.parse_args()

    # 2. Configura o logging
    setup_logging(debug=args.debug)

    # 3. Inicia a aplicação
    app = QApplication(sys.argv)
    
    try:
        from PyQt6.QtWebEngineCore import QWebEngineProfile
        QWebEngineProfile.defaultProfile().setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
    except ImportError as e:
        logging.warning(f"QWebEngineCore não encontrado. O visualizador de PDF pode falhar. {e}")

    # Passa o estado de debug para a MainWindow
    window = MainWindow(debug_mode=args.debug) 
    window.show()
    sys.exit(app.exec())
