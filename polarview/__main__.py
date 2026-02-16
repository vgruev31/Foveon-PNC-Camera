"""Entry point for ``python -m polarview``."""

import sys

from PyQt6.QtWidgets import QApplication

from .main_window import PolarViewMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = PolarViewMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
