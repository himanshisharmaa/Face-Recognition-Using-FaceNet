from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from numpy import array
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)

from tensorflow.keras.models import load_model
from numpy import expand_dims
from numpy import reshape
from numpy import load
from numpy import max
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import cv2
import os



#Method to extract face
def extract_image(image):
    #open the image
    img=Image.open(image)
    img=img.convert('RGB')
    # img1=cv2.imread(image)
    pixels=asarray(img)
    detector=MTCNN()
    f=detector.detect_faces(pixels)
    # print(f)
    # # print(f[0]['box'])
    x1,y1,w,h=f[0]['box']
    # output=img1[y1:y1+h,x1:x1+w]
    # cv2.imshow("Original",img1)
    # cv2.imshow("Output",output)
    # cv2.waitKey()
    x1,y1=abs(x1),abs(y1)
    x2=abs(x1+w)
    y2=abs(y1+h)
    store_face=pixels[y1:y2,x1:x2]
    image1=Image.fromarray(store_face,'RGB')
    image1=image1.resize((160,160))
    face_array=asarray(image1)
    return face_array

# extracting embeddings
def extract_embeddings(model,face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean=face_pixels.mean()
    std=face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=expand_dims(face_pixels,axis=0)
    yhat=model.predict(samples)
    print("Predictions",yhat)
    return yhat[0]

#load the data and reshape the image   
img='E:\DeepLearning\FaceDetection\Face Recognition\FaceNet\Indian-celebrities\Aamir_Khan\Aamir_Khan_1.jpg' 
face=extract_image(img)
testX=asarray(face)
testX=testX.reshape(-1,160,160,3)


print("Input test data shape: ",testX.shape)

# find embeddings
model=load_model("facenet_keras.h5")
model.save("facenet_saved_model", save_format="tf")
model = tf.keras.models.load_model("facenet_saved_model")


new_testx=list()
for test_pixels in testX:
    embeddings=extract_embeddings(model,test_pixels)
    new_testx.append(embeddings)
new_testx=asarray(new_testx)
print("Input test embedding shape: ",new_testx.shape)

data = load('./Compressed_files/Indian-celeb-dataset.npz')
train_x,train_y= data['arr_0'],data['arr_1']

data = load('./Compressed_files/Indian-celeb-embeddings.npz')
trainx,trainy= data['arr_0'],data['arr_1']
print("Loaded data: Train=%d , Test=%d"%(trainx.shape[0],new_testx.shape[0]))

#Normalize the input data
in_encode=Normalizer(norm='l2')
trainx=in_encode.transform(trainx)
new_testx=in_encode.transform(new_testx)

#create a label vector
new_testy = trainy 
out_encode = LabelEncoder()
out_encode.fit(trainy)
trainy = out_encode.transform(trainy)
new_testy = out_encode.transform(new_testy)

#define svm classifier model 
model =SVC(kernel='linear', probability=True)
model.fit(trainx,trainy)


#predict
predict_train = model.predict(trainx)
predict_test = model.predict(new_testx)

#get the confidence score
probability = model.predict_proba(new_testx)
confidence = max(probability)

#Accuracy
acc_train = accuracy_score(trainy,predict_train)


#display
trainy_list = list(trainy)
p=int(predict_test)
if p in trainy_list:
  val = trainy_list.index(p)
#display Input Image
plt.subplot(1,2,1)
plt.imshow(face)
predict_test = out_encode.inverse_transform(predict_test)
plt.title(predict_test)
plt.xlabel("Input Image")
#display Predicated data
plt.subplot(1,2,2)
plt.imshow(train_x[val])
trainy = out_encode.inverse_transform(trainy)
plt.title(trainy[val])
plt.xlabel("Predicted Data")

plt.show()