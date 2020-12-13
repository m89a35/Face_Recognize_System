import sys 

import cv2
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit
from PyQt5.QtWidgets import QMainWindow, QPushButton , QWidget , QMessageBox, QHBoxLayout, QTableWidgetItem
from PyQt5.uic import loadUi

from PyQt5.QtCore import QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
import dlib
import cv2
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import time
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils
import time
################################################
# 專案的根目錄路徑
ROOT_DIR = os.getcwd()

## 訓練/驗證用的資料目錄
gui_path = os.path.join(ROOT_DIR, "webcam2 .ui") 
DATA_PATH = os.path.join(ROOT_DIR, "data")
target_dir = os.path.join(ROOT_DIR, "authorized_person")
unknown_dir = os.path.join(ROOT_DIR, "unknown_person")

# 模型資料目錄
MODEL_PATH = os.path.join(ROOT_DIR, "model")

predictor_path = os.path.join(ROOT_DIR, "shape_predictor_68_face_landmarks.dat")
# C:\Users\ChiaRongHsu\Desktop\專題-original\shape_predictor_68_face_landmarks.dat
# predictor_path = os.path.join(ROOT_DIR, "C:\Users\ChiaRongHsu\Desktop\專題-original\shape_predictor_68_face_landmarks.dat")
face_rec_model_path = os.path.join(ROOT_DIR, "dlib_face_recognition_resnet_model_v1.dat")

## nn_model
nn_model_dir = os.path.join(ROOT_DIR, "nn_model")
nn_model_dir = nn_model_dir + "\\"

## 測試用圖像
input_data = os.path.join(DATA_PATH, "video")
image_file = os.path.join(DATA_PATH, "IMG_6529.JPG")
video_file = os.path.join(DATA_PATH, "jokowi_mbek.mp4")

## test
#test_dir = os.path.join(ROOT_DIR, "test")
test = os.path.join(ROOT_DIR, "test.csv")
temp = os.path.join(ROOT_DIR, "temp.csv")

gui_dir = os.path.join(ROOT_DIR, "gui")
gui = os.path.join(gui_dir, "makegui .ui")


if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
#if not os.path.exists(MODEL_PATH):
#        os.makedirs(MODEL_PATH)
        
if not os.path.exists( nn_model_dir ):
        os.makedirs(nn_model_dir)

#if not os.path.exists( test_dir ):
#        os.makedirs(test_dir)
        
if not os.path.exists( gui_dir ):
        os.makedirs(gui_dir)
        


hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'



# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+labeldict_filename)
#print(label_dict)
json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


check = [] # list to check 是否為本人
identity_list = []
check_before_execute = False #第一次抓到人臉時確認是否為自己 
roll_call_file = test # 點名用的csv檔案

first_frame = True #第一張frame

################################################
max_user=10
# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance=-0.5


class Webcam(QDialog):
    def __init__(self):
        super(Webcam,self).__init__()
        
        #讀入設計好的UI檔
        loadUi( gui, self)
        
        
        self.image = None
        
#        #m/連結名為"starButton"的按紐
#        self.startButton.clicked.connect(self.start_webcam)
#        #連結名為"stopButton"的按紐
#        self.stopButton.clicked.connect(self.stop_webcam)
#        
        self.turnButton.setCheckable(True)
        self.turnButton.toggled.connect(self.turn_webcam)
        self.switch_Enable = False 
        #setCheckable()方法，“True”設定該button為可選屬性，及存在“開”和“關”兩種狀態。 
        #setChecked()方法，設定button的狀態為為選中的狀態。
        self.detectButton.setCheckable(True)
        
        #toggled(),當button的標記狀態發生改變的時候觸發訊號
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled = False
#        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector() 
        self.status = 1
        self.status2 = 1
        self.progressBar.setValue(0)
        
    
    def detect_webcam_face(self, status) :
        if status :
            self.detectButton.setText('Stop Detection')
            self.face_Enabled = True
        else :
            self.detectButton.setText('Detect Face')
            self.face_Enabled = False

    def turn_webcam(self, status2) :
        if status2 :
            self.turnButton.setText('Stop/Pause')
            self.switch_Enable = True
#            self.startButton.setStyleSheet("background-color: red")
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(5)
        else :
            self.turnButton.setText('Start Webcam')
            self.switch_Enable = False
            self.timer.stop()

        
#    def start_webcam(self):
#        self.startButton.setStyleSheet("background-color: red")
#        self.capture = cv2.VideoCapture(0)
#        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#        
#        self.timer = QTimer(self)
#        self.timer.timeout.connect(self.update_frame)
#        self.timer.start(5)
        
    def update_frame(self):
        ret,self.image = self.capture.read()
        faceAlign = self.image.copy()
#        start=time.time()
        
        
        self.image = cv2.flip(self.image,1)   #水平翻轉
        
        if(self.face_Enabled):
            detected_image = self.detect_face(self.image,faceAlign)
            self.displayImage(detected_image,1)
        else :
            self.displayImage(self.image,1)
            

            
    def detect_face(self,img,faceAlign) :
        highest_proba=0
        counter=0
        global identity
        global user_i
        global label_prob
        global check_before_execute
        global first_frame
        global check
        
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dets,scores,idx = self.detector.run(self.image, 0,tolerance)    
#        faces = self.faceCascade.detectMultiScale(gray,1.2,5,minSize = (90,90))
        
#        for(x,y,w,h) in dets :
#            print("yoyoyo")
#            cv2.rectangle(img,(x,y),(x+w, y+h), (0,0,255),2)
        color=(255,0,0)
        for i, d in enumerate(dets):
            cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),color,2)    
            dets, scores, idx = self.detector.run(faceAlign, 0, 0)
            for i1, d1 in enumerate(dets):
                shape = sp(faceAlign, d1)  # mark 68_face_landmarks
                face_descriptor = np.array([facerec.compute_face_descriptor(faceAlign, shape)])
                prediction = cnn_model.predict_proba(face_descriptor)
                            

                # print prediction
                for prob in prediction[0]:
                    if prob > highest_proba and prob >=0.1:
                        highest_proba=prob
                        label=counter
                        label_prob=prob
                        identity = label_dict[label]
                    if counter==(len(label_dict)-1) and highest_proba==0: #unknown
                        label= label_dict[counter]
                        label_prob=prob
                        identity=label
                        
                    counter+=1
            if identity!='UNKNOWN':
                if check_before_execute == False :
                    if first_frame == False :
                        first_frame = True
#                        ans = str(input(" T or F ? ") )
                        reply = QMessageBox.question(self, '身分確認', '確認是【'+identity+'】這個學號嗎？', 
                            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
#                        form = WinForm()  
#                        form.show()  
                        if reply == QMessageBox.Yes:
                            user_i = identity
                            self.progressBar.setValue(10)
                            check.append(identity)
                            check_before_execute = True
                        if reply == QMessageBox.Cancel:
                             status = 0
                             self.detect_webcam_face(status)
#                            self.detectButton.setText('Detect Face')
#                            self.face_Enabled = False
                            

#                        if ans == "T" or ans == 't' :
#                            check.append(identity)
#                            check_before_execute = True
                    else :  
                        first_frame = False
#                        cv2.putText(img, "Are you " + identity + " ? ",(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, identity ,(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)                        
                else : 
                    if identity == user_i :
                      self.progressBar.setValue(self.progressBar.value() + 10)
                    if ( identity != user_i ) and ( self.progressBar.value() >= 10 ) :
                      self.progressBar.setValue(self.progressBar.value()- 10)                    
                    check.append(identity)
                    cv2.putText(img, identity ,(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#                    cv2.imshow('face detect', img)
                    if len( check ) >= 15 :
                        self.check_ident(check, user_i)
                        check_before_execute = False
            else :
                cv2.putText(img,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                
        return img
    
    def roll_call(self, roll_call_file, identity ) :  # 點名
     
        with open( roll_call_file, 'r+' , newline='') as csvfile:
      # 讀取 CSV 檔案內容
          dic = csv.DictReader(csvfile)
          with open( "temp.csv", 'w+' , newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dic.fieldnames)
            writer.writeheader()
              #name = str(input('Enter student id : ') )
            name = identity
            num_of_stu = 0
            
            for row in dic:
              num_of_stu = num_of_stu + 1
              if row['學號'] == name :
                  row['未到'] = 'F'
                  row['實到'] = 'T'
                  writer.writerow(row)
          
              else :
                if row['未到'] == 'T':
                    row['未到'] = 'T'
                    row['實到'] = 'F'
                writer.writerow(row)
              print( row )
              self.label.setText("應到人數 : " + str(num_of_stu))
          
          csvfile.close()
        os.remove( roll_call_file )
        os.rename("temp.csv", roll_call_file )        
    
    def check_ident(self, check, identity ) : # 這裡identity是使用者回答的學號
        counter = 0
        global roll_call_file
        global identity_list


        for i in range(len(check)) :
            if identity == check[i] :
                
                counter += 1

        if counter >= 10 :
            self.progressBar.setValue(100)
            QMessageBox.information(self, "恭喜", "辨識成功")
            status = 0
            self.detect_webcam_face(status)
            
            have_i = False  # 確認是否已點過
            
            for i in range(len(identity_list)) :
              if  identity == identity_list[i] :
                  have_i = True
                  break
            
            if have_i == False :
                identity_list.append(identity)
                self.show_number(identity_list) 
            else : 
                QMessageBox.information(self, "提示", "你已經辨識過了喔")
            
            self.label_2.setText("實到人數 : " + str(len(identity_list)))
            #mark in csv file
            self.roll_call( roll_call_file, identity )
#            input(" Press enter to continue ! ")
        check.clear() # init list


    def show_number(self, identity_list) :         
        i = 0 
        self.tableWidget.setRowCount(len(identity_list))
        for i in range(len(identity_list)) :
            self.tableWidget.setItem(0, i, QTableWidgetItem(str(identity_list[i])))

#        self.tableWidget.update()
#        self.frame.update()
        
        
#    def stop_webcam(self):
#        self.timer.stop()
        
        
    
    
    def displayImage(self,img, window = 1 ) :
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3 : #[0] = rows , [1]cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else : 
                qformat = QImage.Format_RGB888
        
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #BGR>>RGB
        outImage = outImage.rgbSwapped()
        
        if window == 1 :
            self.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.imglabel.setScaledContents(True)
    
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window=Webcam()
    window.setWindowTitle('Face Recognition System')
    window.show()
    sys.exit(app.exec_())
