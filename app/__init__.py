from flask import Flask,request,jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO 
import model_set as m

import time
import cv2 as cv
import numpy as np
app=Flask(__name__)
CORS(app)
    
@app.route('/')
def index():
    return 'hello!'
    
@app.route('/predict',methods=['POST'])
def predict():
    faceProto = "model/opencv_face_detector.pbtxt"
    faceModel = "model/opencv_face_detector_uint8.pb"

    ageProto = "model/age_deploy.prototxt"
    ageModel = "model/age_net.caffemodel"

    genderProto = "model/gender_deploy.prototxt"
    genderModel = "model/gender_net.caffemodel"
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    # initialize the data dictionary that will be returned from the
    # view
    data = {'success': False}
    print('request')
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('image'):
            # 從 flask request 中讀取圖片（byte str）
            image = request.files['image'].read()
            image = cv.imdecode(np.frombuffer(image, np.uint8), cv.IMREAD_COLOR)

            padding = 20
            data['predictions'] = []
            # Read frame
            t = time.time()
            frameFace, bboxes = m.getFaceBox(faceNet, image)
            if not bboxes:
                print("No face Detected, Checking next frame")

            for bbox in bboxes:
                face = image[max(0,bbox[1]-padding):min(bbox[3]+padding,image.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, image.shape[1]-1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                # print("Gender Output : {}".format(genderPreds))
                #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
                    
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                #print("Age Output : {}".format(agePreds))
                #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
                r = {'number' : bbox,'gender': gender, 'age': age}
                data['predictions'].append(r)
            data['success'] = True

    return jsonify(data)

@app.route('/getphoto')
def getphoto():
    return 'hi getphoto!'

@app.route('/img/send_img', methods=['POST'])
def send_img():
    f = request.files['file']
    img = f.read()
    print('img',type(img))
    im1 = Image.open(img)
    print('im1',type(im1))
    im2 = cv.imread(img)
    print('im2',type(im2))
    return 'Send Img Test'

if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
        'please wait until server has fully started/n'))
    print("*/n")
    app.run(host='0.0.0.0',port=3000,debug=True)
