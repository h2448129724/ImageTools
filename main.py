"""Entry point for the image processing toolbox."""
import sys
import logging
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
