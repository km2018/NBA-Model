import sys
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class window(QMainWindow):

    #create GUI window with selected specifications
    def __init__(self):
        super(window, self).__init__()
        #size
        self.setGeometry(50, 50, 500, 300)
        #title
        self.setWindowTitle('NBA Predictive Model')
        
        #specify close protocol
        eject = QAction('Quitting', self)
        eject.setShortcut('Ctrl+Q')
        eject.setStatusTip('Application is closing')
        eject.triggered.connect(self.close_application)
        self.statusBar()

        #specify menu bar for GUI and add close option
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(eject)
        
        self.toolBar = self.addToolBar('Close')
        self.toolBar.addAction(eject)
        
        #specify a home screen
        self.home()

    #features on home screen
    def home(self):
        btn = QPushButton('Quit', self)
        btn.clicked.connect(self.close_application)
        
        btn.resize(btn.minimumSizeHint())
        btn.move(0, 100)
        self.show()

    #close GUI
    def close_application(self):
        choice = QMessageBox.question(self, 'Eject', 'Initate Ejection Protocol?', QMessageBox.Yes | 
                                     QMessageBox.No)
        if choice == QMessageBox.Yes:
            print('Ejection completed!')
            sys.exit()
        else:
            pass

#initiate GUI
def run():    
    app = QApplication(sys.argv)
    Gui = window()
    sys.exit(app.exec_())


run()
