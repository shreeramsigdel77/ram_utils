import cv2

#download the file from opencv git
#https://github.com/opencv/opencv/tree/master/data/haarcascades


#loading the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#read input image (loads in BGR format)
img = cv2.imread('sample.jpg')

#convert to grayscale
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces with face_cascade
faceslist = face_cascade.detectMultiScale(grayimg, 1.1,5)  #inputimage scaleFactor and minNeighbours

print(len(faceslist))
#draw rectangles around faces
for (x,y,w,h) in faceslist:
    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

#display
cv2.imshow("Preview",img)
cv2.waitKey(0)
cv2.imwrite("sample_output.png",img)