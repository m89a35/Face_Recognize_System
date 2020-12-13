"""
This script is used to demonstrate face recognition using webcam
"""
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
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import time
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils
# ================================PARAMETER============================================

# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance=-0.5
target_dir='authorized_person/'

# The directory of the trained neural net model
nn_model_dir='nn_model/'
hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_face_recognition_resnet_model_v1.dat'

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+labeldict_filename)
#print(label_dict)
# ====================================================================================

json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    faceAlign = frame.copy()
    start=time.time()

    dets,scores,idx = detector.run(frame, 0,tolerance)

    for i, d in enumerate(dets):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
        faceAlign = fa.align(frame, frame, d)  # new photo
        dets, scores, idx = detector.run(faceAlign, 0, 0)  # new photo 還需要再 detect一次
        for i1, d1 in enumerate(dets):
            shape = sp(faceAlign, d1)
            face_descriptor = np.array([facerec.compute_face_descriptor(faceAlign, shape)])
            prediction = cnn_model.predict_proba(face_descriptor)

            highest_proba=0
            counter=0
            # print prediction
            for prob in prediction[0]:
                if prob > highest_proba and prob >=0.1:
                    highest_proba=prob
                    label=counter
                    label_prob=prob
                    identity = label_dict[label]

                if counter==(len(label_dict)-1) and highest_proba==0: #unknow
                    label= label_dict[counter]
                    label_prob=prob
                    identity=label

                counter+=1

            if identity!='UNKNOWN':
                cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            else:
                cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    #cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('face detect', frame)
    cv2.imshow('face align', faceAlign)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    delta=time.time()-start
    fps=float(1)/float(delta)
    print(int(fps))

# When everything done, release the capture
