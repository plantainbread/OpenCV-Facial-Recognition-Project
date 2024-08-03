import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path="datasets"

def getImageID(path):
    # Grabs all image paths from the datasets directory
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePaths in imagePath:
        #Opens the image in the imagePath list and convert to grayscale
        faceImage = Image.open(imagePaths).convert('L')
        #Convert the grayscale images to arrays
        faceNP = np.array(faceImage)
        #Extracts the ID of the face image by slicing the file name.
        Id= (os.path.split(imagePaths)[-1].split(".")[1])
        Id=int(Id)
        #Append the numpy array representation of the dataset image to a new list called faces.
        faces.append(faceNP)
        #Appends the ID of the corresponding images in a different list.
        ids.append(Id)
        #Display the image as it is being processed as seen above.
        cv2.imshow("Training",faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
# Your model will be stored in a file named Trainer.yml
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training process finished.")
