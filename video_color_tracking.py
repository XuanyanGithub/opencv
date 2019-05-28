import cv2 as cv
import numpy as np
def color_object_trace():
    capture = cv.VideoCapture(0)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv.CAP_PROP_FPS)
    print(height, width, count, fps)
    k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    while (True):
        ret, frame = capture.read()
        if ret is True:
            cv.imshow("video-input", frame)
            # stage one
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #mask = cv.inRange(hsv, (0, 0, 0), (180,255,46))#黑色
            #mask = cv.inRange(hsv, (35, 43, 46), (77, 255, 255))#绿色
            mask = cv.inRange(hsv, (0, 43, 46), (10, 255, 255))#红色
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
            cv.imshow("mask", mask)
            image, contours, hierachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            max = 0
            temp = 0
            index = -1
            for i in range(len(contours)):
                x, y, w, h = cv.boundingRect(contours[i])
                temp = w*h
                if temp > max:
                    max = temp
                    index = i
            if index >= 0:
                rrt = cv.fitEllipse(contours[index])
                cv.ellipse(frame, rrt, (0, 0, 255), 2, cv.LINE_AA)
            cv.imshow("trac-object-demo", frame)

            c = cv.waitKey(50)
            if c == 27:  # ESC
                break
        else:
            break


if __name__ == "__main__":
    color_object_trace()
    cv.waitKey(0)
    cv.destroyAllWindows()