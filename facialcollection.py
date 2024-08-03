import cv2

#turns on the camera
video=cv2.VideoCapture(0)

#face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter your ID: ")
# int(id) <-- Optional
count=0

while True:
    ret,frame = video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(1)

    if count>500 or k == ord('q'):
        break

#stops video
video.release()

#closes video window
cv2.destroyAllWindows()
print("Dataset Collection Done..................")
