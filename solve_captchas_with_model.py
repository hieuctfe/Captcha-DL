from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import random
import string
import tensorflow as tf
from os.path import join
import json
from flask import Flask, render_template
from flask import request
import requests
import base64
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "test"
RAW_PATH = "craw_img"
solve_captcha_path = "solved_captcha"
log_folder = "log/"
log_file = "08_03_2019.txt"

global graph
graph = tf.get_default_graph()
model = load_model(MODEL_FILENAME)

def rand_string(N=6):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def crawl_img(sessionId):
    # url = "http://captcha.alibaba.com/get_img?identity=aliexpress.com&sessionid=" + sessionId
    url = "http://captcha.alibaba.com/get_img?identity=alibaba.com&sessionid=" + sessionId
    randomStr = rand_string()
    file_path = join(RAW_PATH,randomStr+'.jpg')
    with open(file_path, 'wb') as f:
        response = requests.get(url)
        if response.ok: f.write(response.content)
        return file_path

def saveBase64ToFile(base64Str):
    randomStr = rand_string()
    imgdata = base64.b64decode(base64Str)
    filename = join(RAW_PATH,randomStr+'.jpg')
    with open(filename, 'wb') as f:
        f.write(imgdata)
        return filename

def saveSolvedCaptcha(base64Str,captcha):
    imgdata = base64.b64decode(base64Str)
    randomStr = rand_string(9)
    filename = join(solve_captcha_path,randomStr + '_' + captcha+'.jpg')
    with open(filename, 'wb') as f:
        f.write(imgdata)
        return filename

@app.route('/solvedcaptcha', methods=['GET','POST'])
def saveCaptcha():
    try:
        saveSolvedCaptcha(request.form.get('base64Str'), request.form.get('captcha'))
        date = request.form.get('date')
        with open(log_folder + log_file, "a") as myfile:
            myfile.write('{date: "' + date + '", value: "T"},\n')
    except:
        return "False"
    return "True"

@app.route('/increfailcaptcha', methods=['GET','POST'])
def increfailcaptcha():
    try:
        date = request.form.get('date')
        with open(log_folder + log_file, "a") as myfile:
            myfile.write('{date: "' + date + '", value: "F"},\n')
        return "True"
    except:
        return "False"


@app.route('/countsolvedcaptcha', methods=['GET'])
def countsolvedcaptcha():
    try:
        data = ''
        with open(log_folder + log_file, "r") as myfile:
            data = myfile.read()
            return render_template('report_page.html', data = data, start_date = log_file)
    except:
        return "server err"
    # return "number sucess: " + str(len(captcha_image_files)) + "/number fail: " + countFailCaptcha()


@app.route('/clearlogdata', methods=['GET'])
def clearlogdata():
    try:
        open(log_folder + log_file, "w").close()
        return "clear success"
    except:
        return "server err"

@app.route('/countcrawdata', methods=['GET'])
def countcrawdata():
    try:
        captcha_image_files = list(paths.list_images(RAW_PATH))
        captcha_image_files2 = list(paths.list_images(solve_captcha_path))
        return str(len(captcha_image_files)) + "/" + str(len(captcha_image_files2))
    except:
        return "server err"


@app.route('/getcaptchatext', methods=['GET','POST'])
def solve_captcha():
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # loop over the image paths
    # for image_file in captcha_image_files:
        # Load the image and convert it to grayscale
    found = False
    captcha_text = ''
    while found == False:
        # image_file = crawl_img(request.args.get('Id'))
        try:
            name = request.form.get('base64Str')
            # print(name)
        except:
            print('khong parse duoc')
        image_file = saveBase64ToFile(request.form.get('base64Str'))

        de = 0
        image = cv2.imread(image_file)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            return json.dumps({
                "status": False,
                "error": "Convert to grey color fail"
            })
        # Add some extra padding around the image
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = []
        # inside of each one
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            #
            # # Compare the width and height of the contour to detect letters that
            # # are conjoined into one chunk
            # if w / h > 1.7:
            #     # This contour is too wide to be a single letter!
            #     # Split it in half into two letter regions!
            #     half_width = int(w / 2)
            #     letter_image_regions.append((x, y, half_width, h))
            #     letter_image_regions.append((x + half_width, y, half_width, h))
            # else:
            #     # This is a normal letter by itself
            #     letter_image_regions.append((x, y, w, h))
            letter_image_regions.append((x, y, w, h))

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correcly. Skip the image instead of saving bad training data!
        if len(letter_image_regions) != 4:
            de=de+1
            return json.dumps({
                "status": False,
                "error": "Letter < 4"
            })

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # Create an output image and a list to hold our predicted letters
        output = cv2.merge([image] * 3)
        predictions = []

        # loop over the lektters
        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
            # cv2.imshow("Output", letter_image)
            # cv2.waitKey()

            letter_image_temp = letter_image


            # Re-size the letter image to 20x20 pixels to match training data
            try:
                letter_image = resize_to_fit(letter_image, 20, 20)
            except:
                return json.dumps({
                    "status": False,
                    "error": "Fit image size fail"
                })
            # cv2.imshow("Output", letter_image)
            # cv2.waitKey()
            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            global graph
            try:
                with graph.as_default():
                    prediction =  model.predict(letter_image)
                    letter =  lb.inverse_transform(prediction)[0]
                    predictions.append(letter)
            except:
                return json.dumps({
                    "status": False,
                    "error": "get prediction fail"
                })

            # Convert the one-hot-encoded prediction back to a normal letter
            

            # randomstr = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
            # cv2.imwrite("test2/" + letter + "/" + randomstr + ".png", letter_image_temp)


            # draw the prediction on the output image
            # cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Print the captcha's text
        captcha_text = "".join(predictions)
        print(image_file)
        # Show the annotated image
        # cv2.imshow("Output", output)
        # cv2.waitKey()
        found = True
        if found == True:
            return json.dumps({
                "status": True,
                "data": captcha_text
            })
    
    
if __name__=='__main__':
    app.run(host= '0.0.0.0')