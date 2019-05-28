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
    type = 0
    while(True):
        ret, frame = capture.read()
        if ret is True:
            cv.imshow("video-input", frame)
            result = process_frame(frame, type)
            cv.imshow("video-result", result)
            c = cv.waitKey(20)
            if c > 0:
                type = np.abs(c) % 4
                print(type)
            #out.write(frame)
            if c == 27:  #ESC
                break
        else:
            break


def process_frame(frame, type):
    if type == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
        return binary
    if type == 1:
        dst = cv.GaussianBlur(frame, (0, 0), 15)
        return dst
    if type == 2:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = cv.filter2D(frame, -1, kernel)
        aw = cv.addWeighted(im, 2, cv.GaussianBlur(frame, (0, 0), 15), -2, 128)
        return aw
    if type == 3:
        dst = cv.bilateralFilter(frame, 0, 50, 5)
        return dst

if __name__ == "__main__":
    video_io_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()