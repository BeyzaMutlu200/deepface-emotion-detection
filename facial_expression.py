from deepface import DeepFace
import cv2
import time 

CAP = cv2.VideoCapture(0)


if not CAP.isOpened():
    print("camera cant open")
    exit()


FACE_CLASSIFIER = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
   
   RET, FRAME = CAP.read()

   faces = FACE_CLASSIFIER.detectMultiScale(FRAME, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

   
   if not RET:
    print("failed to grab frame")
    exit()


   for (x, y, w, h) in faces:
        face_roi = FRAME[y:y+h, x:x+w]
        result = DeepFace.analyze(face_roi, actions=['emotion'],enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        
        time.sleep(0.2)   
        cv2.putText(FRAME,emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(FRAME, (x, y), (x+w, y+h), (255, 0, 0), 2)

   cv2.imshow('Emotion Detection', FRAME)

   if cv2.waitKey(30) == 27:  
     break

CAP.release()
cv2.destroyAllWindows()

