
from flask import Flask, request, render_template, jsonify
import base64


import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('drawingModelChallenge1.h5')
model.make_predict_function()


app = Flask(__name__)


import cv2
import numpy as np

import matplotlib.pyplot as plt


@app.route('/')
def index():
    return render_template('/index.html')



# load the class_names.txt file, we have 4 different classes

with open('./model/class_names.txt', 'r') as file:
    class_labels = file.read().splitlines()

print(class_labels) #['house', 'eye', 'fish', 'cup']
@app.route('/recognize', methods=['POST'])
def recognize():

    #here we receive the image from the web browser 

    if request.method == "POST":
        print("Receive image and is predicting it")


        #we receive the request and get the image under imagebase64 format and decode it
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)



        image = cv2.imread('temp.jpg')

        #we resize the image into a 28x28 size and convert it into from RGB to GRAY 
        
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_prediction = np.reshape(image_gray, (28,28,1))

        #we also have to normalise the image before parsing into the cnn predict model
        image_prediction = (255 - image_prediction.astype('float')) / 255

        #predicting
        prediction = np.argmax(model.predict(np.array([image_prediction])), axis=-1)

        #since we loaded class_labels from a text file, class with the highest prediction percentage would be returned 
        #and sent back into the web 

        predicted_class = class_labels[prediction[0]]
        #running predict here
        return jsonify({
            'prediction': str(predicted_class),
            'status':True
        })

    return render_template('/index.html')


if __name__ == '__main__':
    app.run(debug=True) 