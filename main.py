import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with_mask_data = []
count = 1
for i in list(os.listdir("Dataset/with_mask")):
    img = cv2.imread("Dataset/with_mask/"+str(i))
    haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = haar_data.detectMultiScale(img)
    if type(face) == np.ndarray:
        count += 1
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50, 50))
        if count<=150:
            with_mask_data.append(face)
        else:
            break
print("t")
without_mask_data = []
count = 1
for i in list(os.listdir("Dataset/without_mask")):
    img = cv2.imread("Dataset/without_mask/"+str(i))
    haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = haar_data.detectMultiScale(img)
    if type(face) == np.ndarray:
        count += 1
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50, 50))
        if count<=150:
            without_mask_data.append(face)
        else:
            break

np.save('with_mask.npy', with_mask_data)
np.save('without_mask.npy', without_mask_data)

with_mask = np.load('with_mask.npy')
with_mask = with_mask.reshape(149, 50*50*3)

without_mask = np.load('without_mask.npy')
without_mask = without_mask.reshape(149, 50*50*3)

x = np.r_[with_mask, without_mask]
labels = np.zeros(x.shape[0])
labels[149:] = 1.0
names = {0:'Mask', 1:'No Mask'}
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size = 0.25)
clf = svm.SVC(kernel = 'linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_COMPLEX
for i in range(0, 10):
    img = cv2.imread(f"test_images/test_image_{i}.jpg")
    face = haar_data.detectMultiScale(img)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
        face = img[y:y+h, x:x+w, :]
        face = cv2.resize(face, (50, 50))
        face = face.reshape(1, -1)
        pred = clf.predict(face)[0]
        n = names[int(pred)]
        cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)
    cv2.imshow("image", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
