import cv2

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(grayscaled_img)

    smiles = smile_detector.detectMultiScale(
        grayscaled_img, scaleFactor=1.7, minNeighbors=20)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # slicing multi-dimensional frame to get only the face
        the_face = frame[y:y+h, x:x+w]

        grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(
            grayscaled_face, scaleFactor=1.7, minNeighbors=20)

        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 255))
        # for (x_, y_, w_, h_) in smiles:
            #cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 3)

    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
