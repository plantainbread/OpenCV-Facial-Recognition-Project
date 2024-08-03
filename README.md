# OpenCV-Facial-Recognition-Project
Fun little program I made to be used later for a raspberry pi project, but I also placed the code here for anyone interested :)

**Algorithm using the cv2 library, os, and python image library to detect and recognize human faces.**
### Be sure to remove dummy.txt file from the datasets folder, it is there just so I can add a folder on Github.

My OpenCV-Python is version 4.8.0.74

*install any necessary packages before running the code, they are stated at imports at the top of the code*

For your own usage first run facialcollection.py to obtain images of the user you want the model to recognize. You may set the amount of photos it takes using the count variable.
The openCV window that pops up may be closed by pressing the 'q' key.

Once the desired amount of image samples are collected, they will be stored in the datasets folder, and labeled with numbers depending on the corresponding user. (If you don't know how many
photos to take, I set mine to 500, of course the more you take the better the model will be at predicting the user)

Now that you have the samples you may go to training.py and run the code, this way you will create a new file named Trainer.yml, which is where the model will be saved for future deployment.

Lastly, you will go to predictions.py, where you will set the names of the user you want labeled in the name_list, always leave the first element empty. You may also alter the confidence level of the model to adjust the sensitivity, (which follows the logic if confid < 95, it actually becomes more accurate at identifying the user of interest)

Because there is a separate function called is_face_recognized, you may incorporate this code on any microcontroller capable of running python in for your own projects that involves identifying a certain user, just use the function to determine whether or not the said user is in frame.

Anyways, sorry for the yapping, I hope there isn't any issue with the code on your end, and goodluck working on whatever you're doing.
