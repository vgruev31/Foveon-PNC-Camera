"""Entry point for ``python -m polarview``."""

import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMessageBox

from .main_window import PolarViewMainWindow


def _excepthook(exc_type, exc_value, exc_tb):
    """Show unhandled exceptions in a message box instead of crashing."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sys.stderr.write(tb_text)
    QMessageBox.critical(None, "Unhandled Exception", tb_text)


def main() -> None:
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    window = PolarViewMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
