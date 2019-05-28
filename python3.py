import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
def do_nothing(value):
    print(value)
def binary_demo():
    src=cv.imread("D:/picture1/lena2.jpg",cv.IMREAD_GRAYSCALE )
    cv.namedWindow("input",cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("Threshold","input",0,255,do_nothing)
    cv.imshow("input",src)
    while True:
        t=cv.getTrackbarPos("Threshold","input")
        ret,dst=cv.threshold(src,t,255,cv.THRESH_BINARY)
        cv.imshow("binary",dst)
        print("threshold value:",ret)
        c=cv.waitKey(10)
        if c==27:
            break


def threshold_segmentation_demo():
    src = cv.imread("D:/picture1/guang.tif", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", src)
    ret, dst = cv.threshold(src, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow("binary",dst)
    print("threshold value:",ret)


def threshold_method_demo():
    src = cv.imread("D:/picture1/guang.tif", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", src)
    dst = cv.adaptiveThreshold(src, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY ,25,10)
    cv.imshow("binary", dst)


def threshold_noise_demo():
    src = cv.imread("D:/picture1/rice.tif")
    cv.imshow("input", src)
    #src=cv.medianBlur(src,5)
    #src=cv.fastNlMeansDenoisingColored(src,None,15, 15, 10, 30)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,dst = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary", dst)


def connected_components_demo():
    src = cv.imread("D:/picture1/rice.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary", binary)
    output=cv.connectedComponentsWithStats(binary,connectivity=8,ltype=cv.CV_32S)
    num_lables=output[0]
    lables=output[1]
    stats=output[2]
    centers=output[3]
    colors=[]
    anotherclors=[]
    for i in range(num_lables):
        b=np.random.randint(0,256)
        g=np.random.randint(0,256)
        r=np.random.randint(0,256)
        cb=255-b
        cg=255-g
        cr=255-r
        colors.append((b,g,r))
        anotherclors.append((cb,cg,cr))
    colors[0]=(0,0,0)
    h, w = binary.shape
    image=np.zeros((h,w,3),dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row,col]=colors[lables[row,col]]
    for i in range(num_lables):
        if i==0:
            continue
        cx,cy=centers[i]
        x,y,width,height,area=stats[i]
        cv.rectangle(image,(x,y),(x+width,y+height),(0,0,255),2,8,0)
        cv.circle(image,(np.int(cx),np.int(cy)),2,anotherclors[i],-1,8,0)
        xx=str(int(cx))+","+str(int(cy))+"_"+str(int(area))
        cv.putText(image,xx,(x+3,y+8),cv.FONT_HERSHEY_PLAIN,0.7,(255,255,255),1,cv.LINE_8)
    cv.imshow("colored labels", image)
    print("total rice:", num_lables - 1)



def find_contours_demo():
    src = cv.imread("D:/picture1/coin.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary", binary)
    image,contours,hierachy=cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    print("total number",len(contours))
    for i in range(len(contours)):
        cv.drawContours(src,contours,i,(0,0,255),2,8)
    cv.imshow("countours-demo",src)


def measure_contours_demo():
    src = cv.imread("D:/picture1/rice.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary", binary)
    image,contours,hierachy=cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    print("total number",len(contours))
    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        arclen=cv.arcLength(contours[i],True)
        x,y,ww,hh=cb=cv.boundingRect(contours[i])
        ratio=np.minimum(ww,hh)/np.maximum(ww,hh)
        if (area>10 or arclen>10) and (ratio>0.8):
            mm=cv.moments(contours[i])
            m00=mm['m00']
            m10=mm['m10']
            m01=mm['m01']
            cx=np.int(m10/m00)
            cy=np.int(m01/m00)
            (x,y),(a,b),degree=cv.fitEllipse(contours[i])
            print(np.int(a),np.int(b),degree)
            cv.circle(src,(cx,cy),2,(255,0,0),-1,8,0)
            print("area:%d,arclen:%d"%(area,arclen))
            cv.drawContours(src,contours,i,(0,0,255),2,8)
            cv.putText(src,str(np.int(degree)),(cx-20,cy-20),cv.FONT_HERSHEY_PLAIN,1,(255,0,0),1,8,0)
    cv.imshow("countours-demo",src)



def distance_demo():
    src = cv.imread("D:/picture1/rice.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary",binary)
    dist=cv.distanceTransform(binary,cv.DIST_L1,3,dstType=cv.CV_8U)
    cv.normalize(dist,dist,0,255,cv.NORM_MINMAX)
    ret,binary=cv.threshold(dist,120,255,cv.THRESH_BINARY)
    cv.imshow("distance-transform",dist)
    cv.imshow("distance-binary",binary)


def point_polygon_test_demo():
    src = cv.imread("D:/picture1/t.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary",binary)
    image,contours,hierachy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    print("total number:",len(contours))
    lut_image=cv.imread("D:/picture1/lut_rainbow.png")
    h1,w1=lut_image.shape[:2]
    lut=np.zeros((1,256,3),dtype=np.uint8)
    for col in range(w1):
        lut[0,col]=lut_image[h1//2,col]
    h,w=src.shape[:2]
    t=0
    d=0
    for row in range(h):
        for col in range(w):
            dist=cv.pointPolygonTest(contours[0],(col,row),True)
            if dist>0:
                if (dist > t):
                    t = dist
                b=lut[0,int(256*dist/24),0]
                g=lut[0,int(256*dist/24),1]
                r=lut[0,int(256*dist/24),2]
                src[row,col]=(b,g,r)
            else:
                if (np.abs(dist) > d):
                    d = abs(dist)
                b=lut[0,255-int(np.abs(256*dist/189)),0]
                g=lut[0,255-int(np.abs(256*dist/189)),1]
                r=lut[0,255-int(np.abs(256*dist/189)),2]
                src[row,col]=(b,g,r)
    print(t,d)
    cv.imshow("ppt-demo",src)

def binary_projection_demo():
    src = cv.imread("D:/picture1/t.tif")
    cv.imshow("input", src)
    src=cv.GaussianBlur(src,(9,9),0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary",binary)
    h,w=binary.shape
    print(w,h)
    x_projection=np.zeros((w),dtype=np.int32)
    y_projection=np.zeros((h),dtype=np.int32)
    x1=0
    x2=0
    y1=0
    y2=0
    for col in range(w):
        count=0
        for row in range(h):
            pv=binary[row,col]
            if pv==255:
                count+=1
        x_projection[col]=count
        if (x_projection[col-1]==0)and(x_projection[col]!=0):
            x1=col-1
        if (x_projection[col-1]!=0)and(x_projection[col]==0):
            x2=col-1
    for row in range(h):
        count=0
        for col in range(w):
            pv=binary[row,col]
            if pv==255:
                count+=1
        y_projection[row]=count
        if (y_projection[row-1]==0)and(y_projection[row]!=0):
            y2=row-1
        if (y_projection[row-1]!=0)and(y_projection[row]==0):
            y1=row-1
    cv.rectangle(src,(x1,y2),(x2,y1),(0,0,255),4,cv.LINE_8,0)
    plt.plot(x_projection, color='r')
    plt.xlim([0,w])
    plt.show()
    plt.plot(y_projection,color='b')
    plt.xlim([0, h])
    plt.show()
    print(x1,x2,y1,y2)
    cv.imshow("new image",src)


def match_shapes_demo():
    src1 = cv.imread("D:/picture1/2.png",cv.IMREAD_GRAYSCALE)
    src2 = cv.imread("D:/picture1/2.2.png",cv.IMREAD_GRAYSCALE)
    ret1,binary1 = cv.threshold(src1,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    ret2,binary2 = cv.threshold(src2,0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU )
    cv.imshow("binary1",binary1)
    cv.imshow("binary2",binary2)
    dist1=cv.distanceTransform(binary1,cv.DIST_L1,3,dstType=cv.CV_8U)
    dist2=cv.distanceTransform(binary2,cv.DIST_L1,3,dstType=cv.CV_8U)
    cv.normalize(dist1,dist1,0,255,cv.NORM_MINMAX)
    cv.normalize(dist2,dist2,0,255,cv.NORM_MINMAX)
    ret,binary11=cv.threshold(dist1,130,255,cv.THRESH_BINARY)
    ret,binary22=cv.threshold(dist2,110,255,cv.THRESH_BINARY)
    cv.imshow("binary11",binary11)
    cv.imshow("binary22",binary22)
    image, contours1, hierachy = cv.findContours(binary11, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    image, contours2, hierachy = cv.findContours(binary22, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mm1=cv.moments(contours1[0])
    mm2=cv.moments(contours2[0])
    hu1=cv.HuMoments(mm1)
    hu2=cv.HuMoments(mm2)
    dist=cv.matchShapes(hu1,hu2,cv.CONTOURS_MATCH_I2,0)
    print("match distance:",dist)



def hough_line_dimo():
    src = cv.imread("D:/picture1/lane_line1.png")
    cv.imshow("input", src)
    image=np.copy(src)
    src=cv.GaussianBlur(src,(9,9),0)
    edges=cv.Canny(src,150,300,apertureSize=3)#边缘提取
    cv.imshow("edges",edges)
    lines=cv.HoughLines(edges,1,np.pi/180,150,None,0,0)
    print(lines)
    if lines is not None:
        for i in range(0,len(lines)):
            rho=lines[i][0][0]
            theta=lines[i][0][1]
            a=math.cos(theta)
            b=math.sin(theta)
            x0=a*rho
            y0=b*rho
            pt1=(int(x0+1000*(-b)),int(y0+a*1000))
            pt2=(int(x0-1000*(-b)),int(y0-a*1000))
            cv.line(src,pt1,pt2,(0,0,255),2,8,0)
    cv.imshow("hough_lines",src)
    linesP=cv.HoughLinesP(edges,1,np.pi/180,100,None,50,10)
    print(linesP)
    if linesP is not None:
        for i in range(0,len(linesP)):
            l=linesP[i][0]
            cv.line(image,(l[0],l[1]),(l[2],l[3]),(0,0,255),2,8,0)
    cv.imshow("hough_linesP",image)

def hough_circle_demo():
    src=cv.imread("D:/picture1/coin.tif")
    cv.imshow("input",src)
    gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    gray=cv.fastNlMeansDenoising(gray,None,15,10,30)
    rows,cols=gray.shape
    print(rows/8)
    circles=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,rows/8,None,param1=100,param2=30,minRadius=10,maxRadius=50)
    print(circles)
    if circles is not None:
        for i in circles[0,:]:
            center=(i[0],i[1])
            raidus=i[2]
            cv.circle(src,center,2,(0,0,255),-1,8,0)
            cv.circle(src,center,raidus,(0,255,255),2,8,0)
    cv.imshow("circles",src)

def erode_dilate_demo():
    src = cv.imread("D:/picture1/morphology.png")
    #cv.imshow("input", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    dst = cv.dilate(binary, k)
    dst1=cv.erode(binary,k)
    cv.imshow("result", dst)
    cv.imshow("result1", dst1)

def open_close_demo():
    src = cv.imread("D:/picture1/morphology.png")
    cv.imshow("input", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, k, iterations=3)
    dst1 = cv.morphologyEx(binary, cv.MORPH_OPEN, k, iterations=3)
    cv.imshow("result", dst)
    cv.imshow("result1", dst1)

def hv_lines_demo():
    src = cv.imread("D:/picture1/hv_lines.png")
    cv.imshow("input", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY| cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    k = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
    k1 = cv.getStructuringElement(cv.MORPH_RECT, (1, 30))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, k, iterations=1)
    dst1 = cv.morphologyEx(binary, cv.MORPH_OPEN, k1, iterations=1)
    result=cv.bitwise_or(dst,dst1)
    cv.imshow("result", result)

def gradient_demo():
    src = cv.imread("D:/picture1/lane_line1.png", cv.IMREAD_GRAYSCALE)
    ret, binary = cv.threshold(src, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    src = binary
    cv.imshow("input", src)
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dimage = cv.dilate(src, k)
    eimage = cv.erode(src, k)
    basic_grad = cv.morphologyEx(src, cv.MORPH_GRADIENT, k)
    internal_grad = cv.subtract(src, eimage)
    external_grad = cv.subtract(dimage, src)
    cv.imshow("basic_grad", basic_grad)
    cv.imshow("internal_grad", internal_grad)
    cv.imshow("external_grad", external_grad)
    xk = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    yk = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
    dx = cv.dilate(src, xk)
    ex = cv.erode(src, xk)
    dy = cv.dilate(src, yk)
    ey = cv.erode(src, yk)
    xx = cv.subtract(dx, ex)
    yy = cv.subtract(dy, ey)
    cv.imshow("x-direction", xx)
    cv.imshow("y-direction", yy)

def more_morphology_demo():
    src = cv.imread("D:/picture1/morphology.png", cv.IMREAD_GRAYSCALE)
    ret, binary = cv.threshold(src, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    src = binary
    cv.imshow("input", src)
    k = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    bh = cv.morphologyEx(src, cv.MORPH_BLACKHAT, k)
    cv.imshow("black hat", bh)
    th = cv.morphologyEx(src, cv.MORPH_TOPHAT, k)
    cv.imshow("top hat", th)
    hmk1 = np.zeros((3, 3), dtype=np.uint8)
    hmk1[2, 2] = 1
    hmk2 = np.zeros((3, 3), dtype=np.uint8)
    hmk2[0, 0] = 1
    hm1 = cv.morphologyEx(src, cv.MORPH_HITMISS, hmk1)
    hm2 = cv.morphologyEx(src, cv.MORPH_HITMISS, hmk2)
    cv.imshow("hit and miss 1", hm1)
    cv.imshow("hit and miss 2", hm2)
    hm = cv.add(hm1, hm2)
    cv.imshow("hit and miss", hm)

if __name__=="__main__":
    more_morphology_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()




