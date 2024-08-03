import cv2

# Turns on the webcam (you may have to change the number depending on which webcam you want to use, if you have more than one)
video=cv2.VideoCapture(0)

# Face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter an integer: ")
count=0
record_num = 500 # Change this if you wish to take more photos

while True:
    ret,frame = video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        # Photos taken will be placed in datasets directory
        cv2.imwrite('datasets/User.' + id + '.'+str(count)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)

    cv2.imshow("Window",frame)

    k=cv2.waitKey(1)
    # Terminates window once the count reaches set number or the 'q' key is pressed
    if count>=record_num or k == ord('q'):
        break

#Sstops video
video.release()

# Closes video window
cv2.destroyAllWindows()
print("Collection process finished.")
