from PyQt5 import QtCore, QtGui, QtWidgets
from painterWindow import PainterWindow

class AeroPaint:
    def __init__(self):
        self.painterWindow = QtWidgets.QMainWindow()
        self.painterUI = PainterWindow()
        self.painterUI.setupUi(self.painterWindow)
        
    def setupUi(self, AeroPaint):
        AeroPaint.setObjectName("AeroPaint")
        AeroPaint.resize(480, 360)
        AeroPaint.setMinimumSize(QtCore.QSize(480, 360))
        AeroPaint.setMaximumSize(QtCore.QSize(480, 360))
        self.centralwidget = QtWidgets.QWidget(AeroPaint)
        self.centralwidget.setObjectName("centralwidget")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setEnabled(True)
        self.logo.setGeometry(QtCore.QRect(90, 60, 300, 80))
        self.logo.setStyleSheet("")
        self.logo.setAlignment(QtCore.Qt.AlignCenter)
        self.logo.setObjectName("logo")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./assets/icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        AeroPaint.setWindowIcon(icon)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 190, 180, 40))
        font = QtGui.QFont()
        font.setFamily("Sawasdee")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        AeroPaint.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AeroPaint)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 480, 20))
        self.menubar.setObjectName("menubar")
        self.menumenu = QtWidgets.QMenu(self.menubar)
        self.menumenu.setObjectName("menumenu")
        self.menuVirtual_Painter = QtWidgets.QMenu(self.menubar)
        self.menuVirtual_Painter.setObjectName("menuVirtual_Painter")
        AeroPaint.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AeroPaint)
        self.statusbar.setObjectName("statusbar")
        AeroPaint.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(AeroPaint)
        self.actionClose.setObjectName("actionClose")
        self.actionStart = QtWidgets.QAction(AeroPaint)
        self.actionStart.setObjectName("actionStart")
        self.menumenu.addAction(self.actionClose)
        self.menuVirtual_Painter.addAction(self.actionStart)
        self.menubar.addAction(self.menumenu.menuAction())
        self.menubar.addAction(self.menuVirtual_Painter.menuAction())
        
        self.AeroPaint = AeroPaint

        self.actionClose.triggered.connect(self.close)
        self.actionStart.triggered.connect(self.start_drawing)
        self.pushButton.clicked.connect(self.start_drawing)

        self.retranslateUi(AeroPaint)
        QtCore.QMetaObject.connectSlotsByName(AeroPaint)

    def retranslateUi(self, AeroPaint):
        _translate = QtCore.QCoreApplication.translate
        AeroPaint.setWindowTitle(_translate("AeroPaint", "AeroPaint"))
        self.logo.setText(_translate("AeroPaint", "<html><head/><body><p><img src=\"./assets/AeroPaintLogo.png\"/></p></body></html>"))
        self.pushButton.setText(_translate("AeroPaint", "Start Drawing"))
        self.menumenu.setTitle(_translate("AeroPaint", "Window"))
        self.menuVirtual_Painter.setTitle(_translate("AeroPaint", "Virtual Painter"))
        self.actionClose.setText(_translate("AeroPaint", "Close"))
        self.actionStart.setText(_translate("AeroPaint", "Start Drawing"))

    def start_drawing(self):
        self.painterUI.start(self.AeroPaint)
    
    def close(self):
        self.painterUI.close()
        self.AeroPaint.close()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AeroPaintWindow = QtWidgets.QMainWindow()
    ui = AeroPaint()
    ui.setupUi(AeroPaintWindow)
    AeroPaintWindow.show()
    sys.exit(app.exec_())
