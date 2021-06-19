from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from cv2 import cv2
from src.VirtualPainter import virtualPainter
import pathlib


class PainterWindow:        
    def setupUi(self, painterWindow):        
        painterWindow.setObjectName("painterWindow")
        painterWindow.resize(1280, 720)
        painterWindow.setMinimumSize(QtCore.QSize(1280, 720))
        painterWindow.setMaximumSize(QtCore.QSize(1280, 720))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./assets/icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        painterWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(painterWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        self.label.setText("")
        self.label.setObjectName("label")
        painterWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(painterWindow)
        self.painterWindow = painterWindow
        QtCore.QMetaObject.connectSlotsByName(painterWindow)
        
        self.videoClose = False
        
        painterWindow.closeEvent = self.closeEvent

    def retranslateUi(self, painterWindow):
        _translate = QtCore.QCoreApplication.translate
        painterWindow.setWindowTitle(_translate("painterWindow", "Virtual Painter"))
    
    def start(self, AeroPaint):
        self.AeroPaint = AeroPaint
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)
        self.cap = cap    
        self.painterWindow.show()
        self.load_video(self.cap)
    
    def closeEvent(self, event):
        reply = self.closeMsgBox()
        self.msgBox.close()
        if reply == 'exit with saving':
            self.closeAndSave(event)
        elif reply == 'exit without saving':
            self.close()
            event.accept()
        elif reply == 'cancel':
            event.ignore()
    
    def load_video(self, cap):
        virtualPainter(QtWidgets.QApplication, self.update, cap, self.videoCloseHandler)
    
    def update(self, image, canvas):
        self.canvas = canvas
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1],frame.shape[0],
                             frame.strides[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
    
    def close(self):
        self.videoClose = True
        self.painterWindow.close()
        self.AeroPaint.close()
    
    def videoCloseHandler(self):
        return self.videoClose
    
    def closeAndSave(self, e):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            self.centralwidget, "Save Image", str(pathlib.Path().absolute()), "Images (.jpg, .jpeg, .png)", options=options
        )
        
        if filename:
            name = ''
            ext = ''
            try:
                filename.index('.')
                ext = filename.split('.')[-1]
                name = '.'.join(filename.split('.')[0:-1])
            except ValueError:
                ext = 'jpg'
                name = filename
            
            if not ext in ['jpg', 'jpeg', 'png']:
                cv2.imwrite(name + '.jpg', self.canvas)
            else:
                cv2.imwrite(name + '.' + ext, self.canvas)
            self.close()
            e.accept()
        else:
            e.ignore()
        
    
    def closeMsgBox(self):        
        messageBox = QMessageBox()
        messageBox.setWindowTitle('Exit')
        messageBox.setText('You are about to exit AeroPaint')
        messageBox.setStandardButtons(QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
        cancel_button = messageBox.button(QMessageBox.Cancel)
        cancel_button.setText('cancel')
        exit_without_saving = messageBox.button(QMessageBox.No)
        exit_without_saving.setText('Exit without saving')
        exit_with_saving = messageBox.button(QMessageBox.Yes)
        exit_with_saving.setText('Exit and save')
        messageBox.exec_()
        self.msgBox = messageBox
        if messageBox.clickedButton() == exit_with_saving:
            return 'exit with saving'
        elif messageBox.clickedButton() == exit_without_saving:
            return 'exit without saving'
        elif messageBox.clickedButton() == cancel_button:
            return 'cancel'