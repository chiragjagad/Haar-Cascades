import cv2

video = cv2.VideoCapture('test1.mp4')

car_tracker = cv2.CascadeClassifier('cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    successful_frame_read, frame = video.read()

    if successful_frame_read:

        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_img)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Cars and Pedestrian Detector', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()
