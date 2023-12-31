import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout

class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        menu_widget = QListWidget()
        menu_str=["Today's catch", "Gist","History","About"]
        for i in range(4):
            item = QListWidgetItem(menu_str[i])
            item.setTextAlignment(Qt.AlignCenter)
            menu_widget.addItem(item)
        
        placeholder="Detection and Prevention of Zero day attack using ML"

        text_widget = QLabel(placeholder)
        button = QPushButton("Start analysing")

        content_layout = QVBoxLayout()
        content_layout.addWidget(text_widget)
        content_layout.addWidget(button)
        main_widget = QWidget()
        main_widget.setLayout(content_layout)

        layout = QHBoxLayout()
        layout.addWidget(menu_widget, 2)
        layout.addWidget(main_widget, 4)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication()

    w = Widget()
    w.show()

    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    sys.exit(app.exec())
