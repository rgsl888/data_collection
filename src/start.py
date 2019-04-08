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
from PyQt5.QtCore import QCoreApplication, QTimer, QUrl, QDateTime, Qt, QFileInfo, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWebEngineWidgets import QWebEngineView

import utils.FaceDetection as fd
import utils.GenFeature as genfeat

userName = ""


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi('../gui/login.ui', self)
        self.okButton.clicked.connect(self.login)
        self.tryaga.clicked.connect(self.tryagain)
        self.exitButton.clicked.connect(QCoreApplication.instance().quit)
        self.label, self.image = fd.face_recon() 
        self.userName.setText (self.label)
    
    def login(self):
        global userName
        userName = self.userName.text()
        if ((self.label == "Unknown") or (self.label != userName)):
            cv.imwrite("../userdata/userimages/"+userName+".jpg", self.image)
            genfeat.generate_features()
            
        self.accept()

    def tryagain(self):
        self.label, self.image = fd.face_recon()
        self.userName.setText (self.label)


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
        self.data_path = "../data/"
        self.math_easy = pd.read_csv (self.data_path+"MathEasy.csv"); 
        self.math_med  = pd.read_csv (self.data_path+"MathMed.csv");
        self.math_hard = pd.read_csv (self.data_path+"MathHard.csv");
        self.nextButton.clicked.connect(self.next)
        self.playback.clicked.connect(self.play)
        self.playback.setEnabled(False)
        self.easy.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.easy.setChecked(True)
        self.hard.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.rand.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.a.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.b.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.c.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.d.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.e.setStyleSheet('QRadioButton.indicator { width: 25px; height: 25px;};')
        self.timer = QTimer(self)
        self.userdata_path = "../userdata/data.csv"
        self.timer.timeout.connect(self.updateFrame)
        if not os.path.exists(self.userdata_path): 
            self.userdata = pd.DataFrame(
                {
                    "username"       : [],
                    "level"          : [],
                    "question"       : [],
                    "answer"         : [],
                    "true_answer_opt": [],
                    "user_choice"    : [],
                    "session_video"  : []

                }
            ) 
        else:
           self.userdata = pd.read_csv (self.userdata_path); 
        self.started = False

        self.userdata_temp = pd.DataFrame(
            {
               "username"       : [],
               "level"          : [],
               "question"       : [],
               "answer"         : [],
               "true_answer_opt": [],
               "user_choice"    : [],
               "session_video"  : [] 
            }
        )
    
    def quit(self):
        
        result = QMessageBox.question(self, 'Message', "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.Yes:
            self.save()
            #self.close()
            sys.exit(0)
        else:
            self.webcamEnabled = True

    def save(self):
        if self.webcamEnabled == True:
            self.stopWebCam()

        self.userdata=self.userdata.append(self.userdata_temp, ignore_index=True)
        self.userdata.to_csv (self.userdata_path, index=False)

    def submit(self):
        self.playback.setEnabled(False)
        self.nextButton.setText("Start")


    def next(self):
        if self.started == False:
            self.started = True
        else:
            self.playback.setEnabled(True)
            correct_choice = 0
            if ((self.a.isChecked ()) and (self.ans == 'A')):
                correct_choice = 1
            elif((self.b.isChecked ()) and (self.ans == 'B')):
                correct_choice = 1
            elif((self.c.isChecked ()) and (self.ans == 'C')):
                correct_choice = 1
            elif((self.d.isChecked ()) and (self.ans == 'D')):
                correct_choice = 1
            elif((self.e.isChecked ()) and (self.ans == 'E')):
                correct_choice = 1
            
            self.userdata_temp=self.userdata_temp.append(pd.DataFrame ([[self.user, self.level, self.sample['question'].values[0], self.sample['answer'].values[0], self.ans, correct_choice, self.outputfile]], columns=['username', 'level', 'question', 'answer', 'true_answer_opt', 'user_choice', 'session_video']), ignore_index=True)
            
            
        if self.webcamEnabled == True:
            self.stopWebCam()
        self.nextButton.setText("Next")
        category = ''
        random_state = 2 #  0 - Easy, 1-Hard and 2-Disabled
        if (self.rand.isChecked()):
            random_state = random.randint(0,2)
            
        if ((self.easy.isChecked()) or (random_state == 0)):
            self.data = self.math_easy
            self.level = 'E'
        elif ((self.med.isChecked()) or (random_state == 1)):
            self.data = self.math_med
            self.level = 'M'
        elif ((self.hard.isChecked()) or (random_state == 2)):
            self.data = self.math_hard
            self.level = 'H'

        self.sample = self.data.sample(n=1)
        self.data.drop(index=self.sample.index[0], inplace=True)

        self.view = QWebEngineView()
        self.view.setHtml(str(self.sample['question'].values[0]), QUrl.fromLocalFile(os.getcwd()+"/../data/"))
        self.web.setWidget(self.view)
        self.ans = self.sample['answer_opt'].values[0] 

        if self.webcamEnabled == False:
            self.webcamEnabled = True
            self.savedir = "../userdata/videos/"+self.user
       
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            #fourcc = cv.cv.CV_FOURCC('X','V','I','D')
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

    def play(self):
        self.save()
        self.accept()
        

class PlayBackWindow(QDialog):

    def __init__ (self, userdata):
        super(PlayBackWindow, self).__init__()
        loadUi('../gui/playback.ui', self)
        self.setWindowTitle("Playback")
        self.userdata = userdata
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()

        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        wid = QWidget(self)
  
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        wid.setGeometry (100, 20, 0, 0)
        wid.resize(400,400)
        #fileName = "../userdata/videos/rgsl888/20190308_184703.avi"
        #self.mediaPlayer.setMedia( QMediaContent(QUrl.fromLocalFile(QFileInfo(fileName).absoluteFilePath())))
        self.nextButton.setText("Start")
        self.nextButton.clicked.connect(self.next)
        self.exitButton.clicked.connect(self.exitCall)
        self.count = 0
       
    def next(self):
        self.nextButton.setText("Next")
        if (self.userdata.shape[0] <= self.count):
            QMessageBox.information(self, 'Info', "All the sessions has been already shown, please exit", QMessageBox.Ok)
        else: 
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo(self.userdata['session_video'][self.count]).absoluteFilePath())))
            
            if (self.userdata['level'][self.count] == 'E'):
                lvl = 'Easy'
            elif (self.userdata['level'][self.count] == 'M'):
                lvl = 'Medium'
            else:
                lvl = 'Hard'
          
            self.q_view = QWebEngineView()
            self.q_view.setHtml(str(self.userdata['question'][self.count]), QUrl.fromLocalFile(os.getcwd()+"/../data/"))
            self.question.setWidget(self.q_view)
            
            self.a_view = QWebEngineView()
            self.a_view.setHtml(str(self.userdata['answer'][self.count]), QUrl.fromLocalFile(os.getcwd()+"/../data/"))
            self.true_ans.setWidget(self.a_view)

            self.level.setText('Level: ' + lvl)
            if (self.userdata['user_choice'][self.count] == 0):
                ans = 'Wrong'
            else:
                ans = 'Correct'
            self.answer.setText('User choice: ' + ans)
            self.count = self.count + 1 
        
    def exitCall(self):
        result = QMessageBox.question(self, 'Message', "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.Yes:
            sys.exit(0)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

app = QApplication(sys.argv)
loginWindow = Login()
loginWindow.setWindowTitle('Activity Recorder')
loginWindow.setWindowIcon(QIcon('../resources/icon.png'))
loginWindow.show()

if loginWindow.exec_() == QDialog.Accepted:
    window = MainWindow(userName)
    window.show()
    if window.exec_() == QDialog.Accepted:
        player = PlayBackWindow(window.userdata_temp)
        player.show()
    sys.exit(app.exec_())


