from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from mtcnn import MTCNN
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

#instatiate flask app  
app = Flask(__name__, template_folder='./ui')

camera = cv2.VideoCapture(0)
model = load_model('model.h5')
detector=MTCNN()
emotion_labels = ['Angry','Fear','Happy','Neutral', 'Sad', 'Surprise']
result_label = emotion_labels[0]

def detect_face(image):
  faces = detector.detect_faces(image)
  if len(faces) == 0:
    return
  box = faces[0]['box']
  x1,y1,width,height= box
  x2,y2=x1+width,y1+height

  start_point = (x1,y1)
  end_point = (x2,y2)

  color = (255, 0, 0)
  thickness = 2
  image = cv2.rectangle(image, start_point, end_point, color, thickness)
  cropped =  image[y1:y2,x1:x2]
  gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(gray, (48, 48),interpolation=cv2.INTER_AREA)

  if np.sum([resized])!=0:
    input_tensor = resized.astype('float')/255.0
    input_tensor = img_to_array(input_tensor)
    input_tensor = np.expand_dims(input_tensor,axis=0)
    output_vector = model.predict(input_tensor)[0]

    label=emotion_labels[output_vector.argmax()]
    label_position = (x1,y1)
    cv2.flip(image,1)
    cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

  return image

def gen_frames():
    while True:
        success, frame = camera.read() 
        if success:
            frame = detect_face(frame)

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
