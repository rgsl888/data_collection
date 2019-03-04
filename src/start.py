import cv2 as cv
import sys
import csv
import os
import pickle
import random
import pandas as pd
import time


from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import QCoreApplication, QTimer, QUrl, QDateTime
from PyQt5.QtGui import QIcon, QPixmap

userName = ""


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi('../gui/login.ui', self)
        self.okButton.clicked.connect(self.login)
        self.exitButton.clicked.connect(QCoreApplication.instance().quit)
    
    def login(self):
        global userName
        userName = self.userName.text()
        self.accept()


class MainWindow(QDialog):
    def __init__(self, username):
        super(MainWindow, self).__init__()
        loadUi('../gui/mainwindow.ui', self)
        self.setWindowTitle("Activity Recorder")
        self.setWindowIcon(QIcon('../resources/icon.png'))

        self.username.setText("User - "+ username)
        self.user = username
        self.exitButton.clicked.connect(self.quit)
        self.webcamEnabled = False

        self.categories = [x for x in os.listdir('../data/Images')]
        self.nextButton.clicked.connect(self.next)
        self.submitButton.clicked.connect(self.submit)
        self.submitButton.setEnabled(False)
        self.timer = QTimer(self)
        #self.timer.timeout
        self.timer.timeout.connect(self.updateFrame)
        self.data_path = "../userdata/data.csv"
        if not os.path.exists(self.data_path): 
            self.data = pd.DataFrame(
                {
                    "username"  : [],
                    "label1"    : [],
                    "label2"    : [],
                    "true_label": [],
                    "user_choice"   : [],
                    "session_video" : []

                }
            ) 
        else:
           self.data = pd.read_csv (self.data_path); 
        self.started = False
    
    def quit(self):
        if self.webcamEnabled == True:
            self.stopWebCam()

        self.data.to_csv (self.data_path, index=False)
        
        result = QMessageBox.question(self, 'Message', "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.Yes:
            self.pic1.clear()
            self.pic2.clear()
            #self.close()
            sys.exit(0)
        else:
            self.webcamEnabled = True


    def submit(self):
        self.submitButton.setEnabled(False)
        self.nextButton.setText("Start")
        self.pic1.clear()
        self.pic2.clear()


    def next(self):
        if self.started == False:
            self.started = True
        else:
            correct_choice = 0
            if ((self.ans1.isChecked ()) and (self.ans == 1)):
                correct_choice = 1
                
            elif((self.ans2.isChecked ()) and (self.ans == 2)):
                correct_choice = 1

            self.data=self.data.append(pd.DataFrame ([[self.user, self.label1, self.label2, self.true_label, correct_choice, self.output_filename]], columns=['username', 'label1', 'label2', 'true_label', 'user_choice', 'session_video']), ignore_index=True, sort=False)
            
            
        if self.webcamEnabled == True:
            self.stopWebCam()
        self.nextButton.setText("Next")
        image_categories = random.sample(self.categories, 2)
        image1_name = random.sample(os.listdir('../data/Images/'+image_categories[0]), 1)
        image2_name = random.sample(os.listdir('../data/Images/'+image_categories[1]), 1)

        image1 = QPixmap('../data/Images/'+image_categories[0]+'/'+image1_name[0])
        image2 = QPixmap('../data/Images/'+image_categories[1]+'/'+image2_name[0])
        self.pic1.setPixmap(image1)
        self.pic2.setPixmap(image2)

        self.label1 = image_categories[0][10:].replace('_',' ')
        self.label2 = image_categories[1][10:].replace('_',' ')

        if random.randint(0,1):
            self.label.setText("LABEL : " + self.label1)
            self.ans = 1;
            self.true_label = self.label1
        else:
            self.label.setText("LABEL : " + self.label2)
            self.ans = 2;
            self.true_label = self.label2

        if self.webcamEnabled == False:
            self.webcamEnabled = True
            self.savedir = "../userdata/videos/"+self.user
       
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            fourcc = cv.VideoWriter_fourcc('X','V','I','D')
            self.output_filename = time.strftime("%Y%m%d_%H%M%S")+ ".avi"
            self.outputfile = self.savedir+ "/" + self.output_filename
            self.out = cv.VideoWriter(self.outputfile, fourcc, 30.0, (640,480))
            self.capture = cv.VideoCapture(0)
            self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(16);


    def updateFrame(self):
        ret, image = self.capture.read()

        if ret==True:
            image = cv.flip(image,1)
            self.out.write(image)
        else:
            self.stopWebCam()
            QMessageBox.information(self, 'Info', "Error opening webcam", QMessageBox.Ok)
            self.quit()
            
    def stopWebCam(self):
        self.timer.stop()
        self.capture.release()
        cv.destroyAllWindows()
        self.webcamEnabled = False
            

app = QApplication(sys.argv)
loginWindow = Login()
loginWindow.setWindowTitle('Activity Recorder')
loginWindow.setWindowIcon(QIcon('../resources/icon.png'))
loginWindow.show()

if loginWindow.exec_() == QDialog.Accepted:
    window = MainWindow(userName)
    window.show()
    sys.exit(app.exec_())


