import cv2 as cv
font = cv.FONT_HERSHEY_SIMPLEX
cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3 ,minNeighbors=5)
    print(len(faces))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv.putText(frame,'Face',(x, y), font, 1,(255,0,0),3)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
    cv.putText(frame,'Number of Faces :' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)        
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
