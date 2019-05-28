import cv2 as cv
import numpy as np

def blur_demo():
    src=cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input",src)
    dst=cv.blur(src,(10,10))
    cv.imshow("blur image",dst)

def gaussian_blur_demo():
    src=cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input",src)
    dst=cv.GaussianBlur(src,(0,0),5)
    cv.imshow("gaussian_blur image",dst)

def statistics_filters():
    src=cv.imread("D:/picture1/lenanoise2.jpg")
    cv.imshow("input",src)
    dst=cv.erode(src,(3,3))
    cv.imshow("minimum filter",dst)
    dst2=cv.dilate(src,(3,3))
    cv.imshow("maximum filter",dst2)
    dst3=cv.medianBlur(src,9)
    cv.imshow("median filter",dst3)


def add_noise():
    src=cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input",src)
    copy=np.copy(src)
    h,w=src.shape[:2]
    rows=np.random.randint(0,h,5000,dtype=np.int)
    cols = np.random.randint(0, w, 5000, dtype=np.int)
    for i in range(5000):
        if i%2==1:
            src[rows[i],cols[i]]=(255,255,255)
        else:
            src[rows[i], cols[i]] = (0, 0, 0)
    cv.imshow("salt and pepper image",src)
    gnoise=np.zeros(src.shape,src.dtype)
    m=(15,15,15)
    s=(30,30,30)
    cv.randn(gnoise,m,s)
    dst=cv.add(copy,gnoise)
    cv.imshow("gaussian",dst)
    result=cv.fastNlMeansDenoising(dst,None,15,15,25)
    cv.imshow("result",result)
    result1=cv.fastNlMeansDenoisingColored(dst,None,15,20,15,25)
    cv.imshow("result1",result1)


def gradient_demo():
    src = cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input", src)
    robert_x=np.array([[1,0],[0,-1]])
    robert_y = np.array([[0, -1], [1, 0]])
    prewitt_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    sobel_x=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    lap_4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    lap_8= np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # gradx=cv.filter2D(src,cv.CV_32F,sobel_x)
    # grady=cv.filter2D(src,cv.CV_32F,sobel_y)
    gradx=cv.Sobel(src,cv.CV_32F,1,0)
    grady=cv.Sobel(src,cv.CV_32F,0,1)
    gradx=cv.convertScaleAbs(gradx)
    grady=cv.convertScaleAbs(grady)
    edge=cv.filter2D(src,cv.CV_32F,lap_8)
    edge=cv.convertScaleAbs(edge)
    cv.imshow("x-soble",gradx)
    cv.imshow("y-soble",grady)
    grad=cv.add(gradx,grady)
    cv.imshow("grad",grad)
    cv.imshow("edge",edge)

def do_nothing(p):
    pass
def edge_demo():
    src = cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input", src)
    cv.namedWindow("edge",cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("low","edge",100,300,do_nothing)
    while True:
        low=cv.getTrackbarPos("low","edge")
        edge=cv.Canny(src,low,low*2)
        dst=cv.bitwise_and(src,src,mask=edge)
        cv.imshow("edge",dst)
        c=cv.waitKey(10)
        if c==27:
            break


def sharpen_image():
    lap_5=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    lap_9=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    src = cv.imread("D:/picture1/lena2.jpg")
    cv.imshow("input", src)
    dst=cv.filter2D(src,cv.CV_8U,lap_5)
    cv.imshow("output",dst)


def do_nothing(p):
    pass
def unsharp_mask():
    lap_5=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    src = cv.imread("D:/picture1/muzi.tif")
    cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    dst = cv.filter2D(src, -1, lap_5)
    mask=cv.GaussianBlur(src,(3,3),0)
    cv.createTrackbar("alpha", "result", -10, 10, do_nothing)
    cv.createTrackbar("beta", "result", -10, 10, do_nothing)
    cv.createTrackbar("gama", "result", 0, 255, do_nothing)
    while True:
        alpha=cv.getTrackbarPos("alpha", "result")
        beta=cv.getTrackbarPos("beta", "result")
        gama=cv.getTrackbarPos("gama", "result")
        result=cv.addWeighted(dst,alpha,mask,-beta,gama)
        cv.imshow("result",result)
        c=cv.waitKey(10)
        if c==27:
            break


def do_nothing(p):
    pass
def edge_preserve_demo():
    src = cv.imread("D:/picture1/lian2.jpg")
    cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    cv.createTrackbar("fector1", "result", 0, 200, do_nothing)
    cv.createTrackbar("fector2", "result", 0, 100, do_nothing)
    cv.createTrackbar("fector3", "result", 0, 100, do_nothing)
    while True:
        fector1=cv.getTrackbarPos("fector1", "result")
        fector2=cv.getTrackbarPos("fector2", "result")
        fector3=cv.getTrackbarPos("fector3", "result")
        dst=cv.bilateralFilter(src,0,fector1,fector2/5)
        # dst=cv.edgePreservingFilter(src,None,cv.NORMCONV_FILTER,100,0.3)
        # dst=cv.stylization(src,None,100,0.4)
        #dst1,dst2=cv.pencilSketch(src,None,None,fector1,fector2/100,fector3/100)
        cv.imshow("result",dst)
        #cv.imshow("result_gray",dst1)
        c=cv.waitKey(10)
        if c==27:
            break


def do_nothing(value):
    print(value)
def match_template_demo():
    src = cv.imread("D:/picture1/a.tif")
    tpl = cv.imread("D:/picture1/a_template.png")
    cv.namedWindow("match_demo", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    cv.imshow("template", tpl)
    cv.createTrackbar("fector1", "result", 0, 200, do_nothing)
    cv.createTrackbar("method","match_demo",0,5,do_nothing)
    th,tw=tpl.shape[:2]
    while True:
        method=cv.getTrackbarPos("method","match_demo")
        if method==cv.TM_SQDIFF :
            result=cv.matchTemplate(src,tpl,cv.TM_SQDIFF)
        elif method==cv.TM_SQDIFF_NORMED:
            result=cv.matchTemplate(src,tpl,cv.TM_SQDIFF_NORMED)
        elif method==cv.TM_CCORR:
            result=cv.matchTemplate(src,tpl,cv.TM_CCORR)
        elif method==cv.TM_CCORR_NORMED:
            result=cv.matchTemplate(src,tpl,cv.TM_CCORR_NORMED)
        elif method==cv.TM_CCOEFF:
            result=cv.matchTemplate(src,tpl,cv.TM_CCOEFF)
        else:
            result=cv.matchTemplate(src,tpl,cv.TM_CCOEFF_NORMED)
        minv,maxv,min_loc,max_loc=cv.minMaxLoc(result)
        clone=np.copy(src)
        if method==cv.TM_SQDIFF or method==cv.TM_SQDIFF_NORMED :
            cv.rectangle(clone,min_loc,(min_loc[0]+tw,min_loc[1]+th),(0,0,255),2,8,0)
        else:
            cv.rectangle(clone,max_loc,(max_loc[0]+tw,max_loc[1]+th),(0,0,255),2,8,0)
        cv.imshow("match_demo",clone)
        cv.normalize(result,result,0,255,cv.NORM_MINMAX)
        result=cv.convertScaleAbs(result)
        cv.imshow("result",result)
        c=cv.waitKey(10)
        if c==27:
            break


def template_llk():
    src=cv.imread("D:/picture1/llk.jpg")
    tpl=cv.imread("D:/picture1/llk_template.png")
    cv.imshow("imput",src)
    cv.imshow("tpl",tpl)
    th,tw=tpl.shape[:2]
    result=cv.matchTemplate(src,tpl,cv.TM_CCORR_NORMED)
    t=0.92
    loc=np.where(result>t)
    print(type(loc),loc,loc[::-1],*loc)
    for pt in zip(*loc[::-1]):
        cv.rectangle(src,pt,(pt[0]+tw,pt[1]+th),(0,0,225),2,8,0)
    cv.imshow("llk_demo",src)

def histogram_demo():
    src1=cv.imread("D:/picture1/lena1.jpg")
    src2=cv.imread("D:/picture1/lena2.jpg")
    src3=np.copy(src2)
    h,w=src3.shape[:2]
    rows=np.random.randint(0,h,5000,dtype=np.int)
    cols = np.random.randint(0, w, 5000, dtype=np.int)
    for i in range(5000):
        if i%2==1:
            src3[rows[i],cols[i]]=(255,255,255)
        else:
            src3[rows[i], cols[i]] = (0, 0, 0)
    cv.imshow("imput1",src1)
    cv.imshow("imput2",src2)
    cv.imshow("imput3",src3)
    hist1=cv.calcHist([src1],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    hist2=cv.calcHist([src2],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    hist3=cv.calcHist([src3],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    dist1=cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)
    dist2=cv.compareHist(hist3,hist2,cv.HISTCMP_BHATTACHARYYA)
    dist3=cv.compareHist(hist3,hist3,cv.HISTCMP_BHATTACHARYYA)
    print(dist1,dist2,dist3)



def back_projiction_deno():
    target=cv.imread("D:/picture1/zu_target.jpg")
    sample=cv.imread("D:/picture1/zu_sample.png")
    cv.imshow("target",target)
    cv.imshow("sample",sample)
    target_hsv=cv.cvtColor(target,cv.COLOR_BGR2HSV)
    sample_hsv=cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    roi_hist=cv.calcHist([sample_hsv],[0,1],None,[4,4],[0,180,0,256])
    cv.normalize(roi_hist,roi_hist,0,256,cv.NORM_MINMAX)
    img_backPrj=cv.calcBackProject([target_hsv],[0,1],roi_hist,[0,180,0,256],1)
    k=np.ones((3,3),dtype=np.uint8)
    dst=cv.erode(img_backPrj,k)
    cv.imshow("backprojection",dst)


def pyramid_demo(image):
    level=3
    temp=image.copy()
    #cv.imshow("input",image)
    pyramid_images=[]
    for i in range(level):
        dst=cv.pyrDown(temp)
        pyramid_images.append(dst)
        #cv.imshow("pyramid_down_"+str(i),dst)
        temp=dst.copy()
    return pyramid_images
def laplation_demo():
    src=cv.imread("D:/picture1/lena2.jpg")
    pyramid_images=pyramid_demo(src)
    level=len(pyramid_images)
    for i in range(level-1,-1,-1):
        if(i-1)<0:
            expand=cv.pyrUp(pyramid_images[i],dstsize=src.shape[:2])
            lpls=cv.subtract(src,expand)+127
            cv.imshow("lpls_"+str(i),lpls)
        else:
            expand=cv.pyrUp(pyramid_images[i],dstsize=pyramid_images[i-1].shape[:2])
            lpls=cv.subtract(pyramid_images[i-1],expand)+127
            cv.imshow("lpls_"+str(i),lpls)


def multiple_scale_template_math():
    target=cv.imread("D:/picture1/traffic2.jpg")
    tpl=cv.imread("D:/picture1/traffic3.png")
    cv.imshow("target",target)
    cv.imshow("template",tpl)
    m_target=pyramid_demo(target)
    m_tpl=pyramid_demo(tpl)
    count1=len(m_target)
    count2=len(m_tpl)
    for i in range(count1):
        for j in range(count2):
            temp=m_target[i]
            tpl=m_tpl[j]
            t1,t2=temp.shape[:2]
            t3,t4=tpl.shape[:2]
            if (t1<t3)or(t2<t4):
                continue
            result=cv.matchTemplate(temp,tpl,cv.TM_CCOEFF_NORMED)
            t=0.693
            loc=np.where(result>t)
            for pt in zip(*loc[::-1]):
                cv.rectangle(temp,pt,(pt[0]+t3,pt[1]+t4),(0,0,255),2,8,0)
                cv.imshow("get_match",temp)
            #     found=True
            #     break
            # if found is True:
            #     break

if __name__ =="__main__":
    gradient_demo()
    cv.waitKey(0)
    cv.destroyAllWindows()
