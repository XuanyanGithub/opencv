import cv2 as cv
import numpy as np
def video_io_demo():
    capture = cv.VideoCapture("D:/video1/fbb.mp4")
    # capture = cv.VideoCapture(0)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv.CAP_PROP_FPS)
    print(height, width, count, fps)
    #out = cv.VideoWriter("D:/test.mp4", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15, (np.int(width), np.int(height)), True)

    while(True):
        ret, frame = capture.read()
        if ret is True:
            cv.imshow("video-input", frame)
            result =detect_face(frame)
            cv.imshow("video-result", result)
            c = cv.waitKey(20)
            #out.write(frame)
            if c == 27:  #ESC
                break
        else:
            break

def detect_face(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.equalizeHist(gray, gray)
    faces = face_detector.detectMultiScale(gray, 1.2, 1, minSize=(40, 40), maxSize=(300, 300))
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2, 8, 0)
    return frame

if __name__ == "__main__":
    face_detector = cv.CascadeClassifier("E:/Python Worker/opencv-3.4/data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    #face_detector = cv.CascadeClassifier("E:/Python Worker/opencv-master/data/lbpcascades/lbpcascade_frontalface_improved.xml")
    video_io_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()