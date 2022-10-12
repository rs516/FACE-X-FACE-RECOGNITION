import warnings

import cv2

warnings.filterwarnings(action='ignore')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    global cropped_faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return None
    for (x, y, w, h) in faces:
        cropped_faces = img[y:y + h, x:x + w]

    return cropped_faces


CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

while True:
    ret, frame = CAP.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = 'C:/Users/ROHIT BHATTACHARYYA/OneDrive/Desktop/Images/photo' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
        print("Face found")
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count == 10:
        break
CAP.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!!")
