import sys
import os
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for, session
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time
import h5py
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.debug = True
app.secret_key = "cataract_detection_secret_key_123"


log_pickle_model = pickle.load(open("log_model.sav", 'rb'))
sift = cv2.SIFT_create()

size=128
intensity_model = load_model('squeezenet.h5')


RESULT_DESCRIPTIONS = {
    "Mild Cataract": "Mild cataract is the early stage of lens clouding. It often causes slight blurring or glare, but doesn't usually require immediate surgery. \n\n**How to Cure:** Regular monitoring and updating eyeglass prescriptions are generally recommended at this stage to maintain clear vision. Increasing lighting for reading and using anti-glare sunglasses can also help manage symptoms.",
    "Normal Cataract": "Normal (Moderate) cataract is a progression where the lens clouding becomes more apparent. You may experience increased difficulty with night driving, faded colors, and a more significant impact on daily activities. \n\n**How to Cure:** Consultation with an ophthalmologist is advised to discuss potential surgical options. Cataract surgery is a safe and common procedure where the cloudy lens is replaced with a clear artificial lens (IOL).",
    "Severe Cataract": "Severe cataract represents advanced lens clouding that significantly impairs vision. It can lead to near-total loss of transparency, making simple tasks extremely difficult. \n\n**How to Cure:** At this stage, cataract surgery is usually the most effective and necessary treatment to restore vision and improve quality of life. Modern surgery is highly effective and can often restore vision to near-normal levels.",
    "Normal eye": "Analysis shows your eye is currently healthy with no detectable cataract symptoms. The lens remains clear, allowing light to focus properly on the retina. \n\n**Maintenance:** Continue regular eye check-ups once every 1-2 years and protect your eyes from excessive UV exposure with quality sunglasses."
}


@app.route("/")
def hello():
  return render_template('./home.html')



@app.route("/main/")
def main():
  return render_template('./main.html')

@app.route('/upload_file/', methods = ['POST'])
def upload_file():
    UPLOAD_FOLDER = './static/inputimages'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('main.html', user_image = filename)


@app.route('/prediction/<file>', methods=['GET', 'POST'])
def prediction(file):
  result1 = ""
  result2 = ""
  start = time.time()
  image = cv2.imread("./static/inputimages/"+file,0)
  image_test = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
  glcm_test=[]
  images_sift_test=[]
  img_arr_test = np.array(image_test)
  gCoMat = graycomatrix(img_arr_test, [1], [0],256,symmetric=True, normed=True) # Co-occurance matrix
  contrast = graycoprops(gCoMat, prop='contrast')[0][0]
  dissimilarity = graycoprops(gCoMat, prop='dissimilarity')[0][0]
  homogeneity = graycoprops(gCoMat, prop='homogeneity')[0][0]
  energy = graycoprops(gCoMat, prop='energy')[0][0]
  correlation = graycoprops(gCoMat, prop='correlation')[0][0]
  keypoints, descriptors = sift.detectAndCompute(image_test,None)
  descriptors=np.array(descriptors)
  descriptors=descriptors.flatten()
  glcm_test.append([contrast,dissimilarity,homogeneity,energy,correlation])
  glcm_test=np.array(glcm_test)
  images_sift_test.append(descriptors[:2304])
  images_sift_test=np.array(images_sift_test)
  images_sift_glcm_test=np.concatenate((images_sift_test,glcm_test),axis=1)
  if(log_pickle_model.predict(images_sift_glcm_test)==1):
      result1 = "Cataract detected"
      filename = "./static/inputimages/"+file
      img = cv2.imread(filename)
      img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
      frame = img
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      sensitivity = 156
      lower_white = np.array([0,0,255-sensitivity])
      upper_white = np.array([255,sensitivity,255])
      # Threshold the HSV image to get only white colors
      mask = cv2.inRange(hsv, lower_white, upper_white)
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(frame,frame, mask= mask)
      ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
      circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 100000,param1=80,param2=40,minRadius=0,maxRadius=0)
      x,y,r = 0,0,0
      if circles is not None:
          circles = np.uint16(np.around(circles))
          x,y,r = circles[0][0]
          x=int(x)
          y=int(y)
          r=int(r)
      mask = np.zeros((224,224), np.uint8)
      cv2.circle(mask,(x,y),r,(255,255,255),-1)
      masked_data = cv2.bitwise_and(frame, frame, mask=mask)
      _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
      cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
      x,y,w,h = cv2.boundingRect(cnt[0])

      # Crop masked_data
      crop = masked_data[y:y+h,x:x+w]
      crop = cv2.resize(crop, (224,224), interpolation = cv2.INTER_AREA)
      #preprocess the image
      my_image = preprocess_input(crop)
      crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
      my_image = img_to_array(crop)
      my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
      ans = intensity_model.predict(my_image)
      ans_class = np.argmax(ans)
      classes = ["Mild Cataract","Normal Cataract","Severe Cataract"]
      result2 = classes[ans_class]
  else:
      result1 = "No catarcat"
      result2 = "Normal eye"
  session['result1'] = result1
  session['result2'] = result2
  session['description'] = RESULT_DESCRIPTIONS.get(result2, "No detailed information available.")
  session['user_image'] = file
  return redirect(url_for('results'))

@app.route('/result/', methods=['GET', 'POST'])
def results():
  result1 = session.get('result1', "")
  result2 = session.get('result2', "")
  description = session.get('description', "")
  user_image = session.get('user_image', "")
  return render_template('result.html', result1 = result1, result2 = result2, description = description, user_image = user_image)

if __name__ == "__main__":
  app.run()