import cv2 as cv
import numpy as np
def curve_fitness_demo():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    x = np.array([30, 50, 100, 120])
    y = np.array([100, 150, 240, 200])
    for i in range(len(x)):
        cv.circle(image, (x[i], y[i]), 3, (255, 0, 0), -1, 8, 0)
    cv.imshow("image", image)
    poly = np.poly1d(np.polyfit(x, y, 3))
    print(poly)
    for t in range(30, 250, 1):
        y_ = np.int(poly(t))
        cv.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
    cv.imshow("fit curve", image)

def circle_fitness_demo():
    src = cv.imread("D:/picture1/circle.png")
    cv.imshow("input", src)
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    image, contours, hierachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        rrt = cv.fitEllipse(contours[i])
        cv.ellipse(src, rrt, (0, 0, 255), 2, cv.LINE_AA)
        # x, y = rrt[0]
        # a, b = rrt[1]
        # r = np.int((a/2 + b/2)/2)
        # cv.circle(src, (np.int(x), np.int(y)), r, (255, 0, 255), 2, 8, 0)
        # cv.circle(src, (np.int(x), np.int(y)), 4, (255, 0, 0), -1, 8, 0)
    cv.imshow("fit circle", src)

def line_fitness():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    xp = np.array([30, 50, 100, 120])
    yp = np.array([100, 150, 240, 200])
    h, w, ch = image.shape
    pts = []
    for i in range(len(xp)):
        cv.circle(image, (xp[i], yp[i]), 3, (255, 0, 0), -1, 8, 0)
        pts.append((xp[i], yp[i]))
    cv.imshow("image", image)
    [vx, vy, x, y] = cv.fitLine(np.array(pts), cv.DIST_L1, 0, 0.01, 0.01)
    y1 = int((-x * vy / vx) + y)
    y2 = int(((w - x) * vy / vx) + y)
    cv.line(image, (w - 1, y2), (0, y1), (0, 0, 255), 2)
    cv.imshow("line-fitness", image)


if __name__=="__main__":
    line_fitness()
    cv.waitKey(0)
    cv.destroyAllWindows()