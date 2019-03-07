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

        self.easy_categories = [x for x in os.listdir('../data/easy/')]
        self.hard_categories = [x for x in os.listdir('../data/hard/')]
        self.nextButton.clicked.connect(self.next)
        self.submitButton.clicked.connect(self.submit)
        self.submitButton.setEnabled(False)
        self.timer = QTimer(self)
        #self.timer.timeout
        self.timer.timeout.connect(self.updateFrame)
        self.userdata_path = "../userdata/data.csv"
        if not os.path.exists(self.userdata_path): 
            self.userdata = pd.DataFrame(
                {
                    "username"  : [],
                    "level"     : [],
                    "image1"    : [],
                    "image2"    : [],
                    "label1"    : [],
                    "label2"    : [],
                    "true_label": [],
                    "user_choice"   : [],
                    "session_video" : []

                }
            ) 
        else:
           self.userdata = pd.read_csv (self.userdata_path); 
        self.started = False

        self.userdata_temp = pd.DataFrame(
            {
                "username"  : [],
                "level"     : [],
                "image1"    : [],
                "image2"    : [],
                "label1"    : [], 
                "label2"    : [], 
                "true_label": [], 
                "user_choice"   : [],
                "session_video" : []
                
            }
        )
    
    def quit(self):
        if self.webcamEnabled == True:
            self.stopWebCam()

        self.userdata=self.userdata.append(self.userdata_temp, ignore_index=True, sort=False)
        self.userdata.to_csv (self.userdata_path, index=False)
        
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

            self.userdata_temp=self.userdata_temp.append(pd.DataFrame ([[self.user, self.level, self.image1_path, self.image2_path, self.label1, self.label2, self.true_label, correct_choice, self.outputfile]], columns=['username', 'level', 'image1', 'image2', 'label1', 'label2', 'true_label', 'user_choice', 'session_video']), ignore_index=True, sort=False)
            
            
        if self.webcamEnabled == True:
            self.stopWebCam()
        self.nextButton.setText("Next")
        category = ''
        random_state = 2 #  0 - Easy, 1-Hard and 2-Disabled
        if (self.rand.isChecked()):
            random_state = random.randint(0,1)
            
        if ((self.easy.isChecked()) or (random_state == 0)):
            data_path = "../data/easy/"
            self.level = 'E'
            image_categories = random.sample(self.easy_categories, 2)
        elif ((self.hard.isChecked()) or (random_state == 1)):
            data_path = "../data/hard/"
            self.level = 'H'
            main_category = random.sample(self.hard_categories, 1)
            category = main_category[0]
            sub_categories = [x for x in os.listdir(data_path+main_category[0]+'/')]
            image_categories = random.sample(sub_categories, 2)

        image1_name = random.sample(os.listdir(data_path+category+'/'+image_categories[0]), 1)
        image2_name = random.sample(os.listdir(data_path+category+'/'+image_categories[1]), 1)

        self.image1_path = data_path+category+'/'+image_categories[0]+'/'+image1_name[0] 
        self.image2_path = data_path+category+'/'+image_categories[1]+'/'+image2_name[0]
        image1 = QPixmap(self.image1_path)
        image2 = QPixmap(self.image2_path)
        self.pic1.setPixmap(image1)
        self.pic2.setPixmap(image2)

        self.label1 = image_categories[0][10:].replace('-',' ')
        self.label2 = image_categories[1][10:].replace('-',' ')

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


