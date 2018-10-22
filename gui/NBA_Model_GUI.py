import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class window(QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('pyqt5 Tut')
        # self.setWindowIcon(QIcon('pic.png'))
        
        eject = QAction('&Mayday!', self)
        eject.setShortcut('Ctrl+Q')
        eject.setStatusTip('We\'re going down!')
        eject.triggered.connect(self.close_application)
        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(eject)
        
        self.toolBar = self.addToolBar('Ejection')
        self.toolBar.addAction(eject)
        
        self.home()

    def home(self):
        btn = QPushButton('Quit', self)
        btn.clicked.connect(self.close_application)
        
        btn.resize(btn.minimumSizeHint())
        btn.move(0, 100)
        self.show()

    def close_application(self):
        choice = QMessageBox.question(self, 'Eject', 'Initate Ejection Protocol?', QMessageBox.Yes | 
                                     QMessageBox.No)
        if choice == QMessageBox.Yes:
            print('Ejection completed!')
            sys.exit()
        else:
            pass


def run():    
    app = QApplication(sys.argv)
    Gui = window()
    sys.exit(app.exec_())


run()
