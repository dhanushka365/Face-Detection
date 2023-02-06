import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt.xml")
max_face = 5

while True:
	ret,temp_frame = cap.read()
	frame = cv2.flip(temp_frame,1)
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	
	for face in faces[:max_face]:
		x,y,w,h = face

		offset = 10
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_offset,(100,100))

		cv2.imshow("Face", face_selection)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

	cv2.imshow("faces",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()