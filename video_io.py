import cv2 as cv
import numpy as np
def video_io_demo():
    capture = cv.VideoCapture("D:/video1/LaneLineDetection.mp4")
    #capture = cv.VideoCapture(0)
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
            # out.write(frame)
            c = cv.waitKey(20)
            if c == 27:  #ESC
                break
        else:
            break

if __name__ == "__main__":
    video_io_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()