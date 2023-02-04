import cv2 as cv

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)




face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_eye.xml')
smileCascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_smile.xml')

img = cv.imread('tools\image7.jpg')
if(img is not None):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))


    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        eye_gray = gray[y:y+h, x:x+w]
        eye_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eye_gray)
        smile = smileCascade.detectMultiScale(eye_gray,scaleFactor= 1.3 , minNeighbors=125,minSize=(25, 25),flags=cv.CASCADE_SCALE_IMAGE)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)
        for (sx, sy, sw, sh) in smile:
            cv.rectangle(eye_color, (sh, sy), (sx+sw, sy+sh), (255, 255, 0), 3)
        


    #imS = cv.resize(img, (960, 540))
    img = ResizeWithAspectRatio(img, width=1480) # Resize by width OR    
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

