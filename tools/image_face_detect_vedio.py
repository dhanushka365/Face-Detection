import cv2 as cv
font = cv.FONT_HERSHEY_SIMPLEX
cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye.xml')
eye_glass_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')
# Create our body classifier
body_classifier = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3 ,minNeighbors=6)
    print(len(faces))
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv.putText(frame,'Face',(x, y), font, 1,(255,0,0),3)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(eye_gray, scaleFactor=1.3 ,minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

        eyes_glass = eye_glass_cascade.detectMultiScale(eye_gray, scaleFactor=1.25 ,minNeighbors=5)
        for (ex, ey, ew, eh) in eyes_glass:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

        Smile_rect = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=30 )
        for (x, y, w, h) in Smile_rect:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 150), thickness=2) 
        # Extract bounding boxes for any bodies identified
        for (x,y,w,h) in bodies:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)    

    cv.putText(frame,'Number of Faces :' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)        
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()




"""
scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
Basically the scale factor is used to create your scale pyramid. More explanation can be found here.
 In short, as described here, your model has a fixed size defined during training, which is visible in the xml. 
 This means that this size of face is detected in the image if present. However, by rescaling the input image, 
 you can resize a larger face to a smaller one, making it detectable by the algorithm.
1.05 is a good possible value for this, which means you use a small step for resizing, i.e.
 reduce size by 5%, you increase the chance of a matching size with the model for detection is found.
  This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster
   detection, with the risk of missing some faces altogether.
minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
This parameter will affect the quality of the detected faces. Higher value results in less detections but with 
higher quality. 3~6 is a good value for it.
"""