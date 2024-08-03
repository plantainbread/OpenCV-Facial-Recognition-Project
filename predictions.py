import cv2
import time

def is_face_recognized(snum, confid, name_list):
    if confid < 95 and name_list[snum] != "":
        return True
    else:
        return False

video=cv2.VideoCapture(0)

findFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("Trainer.yml")

name_list = ["", "Ziyao"]

print_interval = 3

last_print_time = time.time()

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = findFace.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        snum, confid = recognizer.predict(gray[y:y+h, x:x+w])
        recognized = is_face_recognized(snum, confid, name_list)
        if recognized:
        #if confid < 95:
            cv2.putText(frame, name_list[snum], (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 227, 0), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (75, 227, 0), 1)
        else:
            cv2.putText(frame, "Unavailable", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 227, 0), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (75, 227, 0), 1)

    cv2.imshow("Frame",frame)

    current_time = time.time()

    if current_time - last_print_time >= print_interval:
        print(f"Face Recognized {recognized}")
        recognized = False
        last_print_time = current_time

    k=cv2.waitKey(1)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Exit...................")