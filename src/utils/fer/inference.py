import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import utils.fer.vgg as vgg

import cv2 as cv

def inference (video_file):
    with_gpu = True
    net = vgg.VGG('VGG19')
    if with_gpu:
        checkpoint = torch.load('./utils/fer/test_model.t7', map_location="cuda:0")
    else:
        checkpoint = torch.load('./utils/fer/test_model.t7', map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    if with_gpu:
        device = torch.device("cuda:0")
        net = net.to(device)
    net.eval()

    face_cascade = cv.CascadeClassifier('/home/rgsl888/.local/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    label2emotion = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt", "None", "Uncertain", "Non-Face"]
    font = cv.FONT_HERSHEY_SIMPLEX
    cap = cv.VideoCapture(video_file)

    now = time.time()
    while(True):
     
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, 
                                          minSize=(100, 100), flags = cv.CASCADE_SCALE_IMAGE)
        if(len(faces)):
            (x,y,w,h) = faces[0]
            face = frame[y:y+h,x:x+w]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
            final_face = cv.resize(face, (300,300), cv.INTER_CUBIC)
    
            input_face = final_face.astype(np.float32)
            input_face = input_face/255.0
            input_face = np.expand_dims(input_face, axis=0)
            input_face = np.transpose(input_face, (0,3,1,2))
            input_face = torch.FloatTensor(input_face)
            if with_gpu:
                input_face = input_face.to(device)
            logits = net(input_face)
            c=int(torch.argmax(logits,1))
            prob = F.softmax(logits[0], dim=0)*100.0
            cv.putText(frame,label2emotion[c],(100,50), font, 2,(0,0,255),2,cv.LINE_AA)
            cv.putText(frame,"Neutral %d"%prob[0],(20,100), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Happy %d"%prob[1],(20,150), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Sad %d"%prob[2],(20,200), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Surprise %d"%prob[3],(20,250), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Fear %d"%prob[4],(20,300), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Disgust %d"%prob[5],(20,350), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Anger %d"%prob[6],(20,400), font, 1,(255,255,255),2,cv.LINE_AA)
            cv.putText(frame,"Contempt %d"%prob[7],(20,450), font, 1,(255,255,255),2,cv.LINE_AA)

            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv.destroyAllWindows()
