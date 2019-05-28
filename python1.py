import cv2 as cv
# import numpy as np
# from matplotlib import pylab as plt
#read and write
src=cv.imread("D:/picture1/lena2.jpg")
#cv.namedWindow("input", cv.WINDOW_AUTOSIZE )
cv.imshow("input", src)
#cv.imwrite("D:/picture2/lena2.png", src)
cv.waitKey(0)
cv.destroyAllWindows()


# draw imge
# src2 = np.zeros([600, 600, 3], dtype=np.uint8)
# for i in range(100000):
#     x1 = np.int(np.random.rand() * 600)
#     y1 = np.int(np.random.rand() * 600)
#     x2 = np.int(np.random.rand() * 600)
#     y2 = np.int(np.random.rand() * 600)
#     b = np.random.randint(0, 256)
#     g = np.random.randint(0, 256)
#     r = np.random.randint(0, 256)
#     cv.line(src2, (x1,y1), (x2,y2),(b,g,r),4,cv.LINE_8,0)
#     cv.imshow("random lines", src2)
#     c=cv.waitKey(60)
#     if c==27:
#         break


#mouse callback
# def mouse_demo():
#     src = np.zeros((512,512,3),dtype=np.uint8)
#     cv.namedWindow("mouse_deno", cv.WINDOW_AUTOSIZE)
#     cv.setMouseCallback("mouse_deno", my_mouse_callback, src)
#     while(True):
#         cv.imshow("mouse_deno",src)
#         c=cv.waitKey(20)
#         if c==27:
#             break
# def my_mouse_callback(event, x, y, flags, params):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(params, (x, y), 50, (0, 0, 255), 2, cv.LINE_8, 0)
# mouse_demo()
# cv.waitKey(0)
# cv.destroyAllWindows()


#print([i for i in dir(cv) if 'EVENT' in i])


#track_bar_demo
# def do_nothing(pr):
#     print(pr)
# def track_bar_demo():
#     src = np.zeros((512, 512, 3), dtype=np.uint8)
#     cv.namedWindow("tb_demo", cv.WINDOW_AUTOSIZE)
#     cv.createTrackbar("B","tb_demo",0,255,do_nothing)
#     cv.createTrackbar("G", "tb_demo", 0, 255, do_nothing)
#     cv.createTrackbar("R", "tb_demo", 0, 255, do_nothing)
#     while(True):
#         b=cv.getTrackbarPos("B","tb_demo")
#         g = cv.getTrackbarPos("G", "tb_demo")
#         r = cv.getTrackbarPos("R", "tb_demo")
#         src[:]=[b,g,r]
#         cv.imshow("tb_demo",src)
#         c = cv.waitKey(20)
#         if c==27:
#             break
# if __name__ == "__main__":
#     track_bar_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# pixel
# def pixel_demo():
#     src = cv.imread("D:/picture1/lena2.jpg")
#     h,w,ch = src.shape
#     print("h,w,ch",h,w,ch)
#     cv.imshow("input",src)
#     for row in range(h):
#         for col in range(w):
#             b,g,r=src[row,col]
#             b=255-b
#             g=255-g
#             r=255-r
#             src[row, col]=[b,g,r]
#     cv.imshow("output",src)
# if __name__ == "__main__":
#     pixel_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# channels: split merge
# def channels_demo():
#     src = cv.imread("D:/picture1/lena2.jpg")
#     cv.imshow("input",src)
#     h,w,ch=src.shape
#     bgr=cv.split(src)
#     cv.imshow("blue",bgr[0])
#     cv.imshow("green", bgr[1])
#     cv.imshow("red", bgr[2])
#     for row in range(h):
#         for col in range(w):
#             b=255-bgr[0][row,col]
#             g = 255 - bgr[1][row, col]
#             r = 255 - bgr[2][row, col]
#             bgr[0][row,col]=b
#             bgr[1][row, col] = b
#             bgr[2][row, col] = b
#     dst=cv.merge(bgr)
#     cv.imshow("dst",dst)
# if __name__ == "__main__":
#     channels_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# flip
# def mirror_demo():
#     src= cv.imread("D:/picture1/lena2.jpg")
#     h, w, ch = src.shape
#     cv.imshow("input", src)
#     dst=np.zeros(src.shape,src.dtype)
#     for row in range(h):
#         for col in range(w):
#             b,g,r=src[row,col]
#             #dst[row,w-col-1]=[b,g,r]#y_flit
#             #dst[h-row-1,col]=[b,g,r]#x_flit
#             dst[h-row-1,w-col-1]=[b,g,r]#xy_flit dst=cv.flip(scr,-1)
#     cv.imshow("flit",dst)
# if __name__ == "__main__":
#     mirror_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


#rotate
# def rotate_demo():
#     src = cv.imread("D:/picture1/lena2.jpg")
#     cv.imshow("input", src)
#     dst=cv.rotate(src,cv.ROTATE_90_CLOCKWISE)
#     cv.imshow("rotate",dst)
#     h,w,ch=src.shape
#     M=cv.getRotationMatrix2D((w//2,h//2),45,1.0)
#     cos=np.abs(M[0,0])
#     sin = np.abs(M[0, 1])
#     nw = np.int(h * sin + w * cos)
#     nh = np.int(h * cos + w * sin)
#     M[0,2]+=(nw/2)-w//2
#     M[1,2]+=(nh/2)-h//2
#     dst2=cv.warpAffine(src,M,(nw,nh))
#     cv.imshow("rotate2",dst2)
# if __name__ == "__main__":
#     rotate_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def resize_image():
#     src = cv.imread("D:/picture1/lena2.jpg")
#     cv.imshow("input", src)
#     h,w=src.shape[:2]
#     print(h,w)
#     dst=cv.resize(src,(w*2,h*2),interpolation=cv.INTER_NEAREST)
#     cv.imshow("dst",dst)
# if __name__ == "__main__":
#     resize_image()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def translate_image():
#     src=cv.imread("D:/picture1/lena2.jpg")
#     cv.imshow("input", src)
#     h,w=src.shape[:2]
#     M=np.float32([[1,0,100],[0,1,100]])
#     dst=cv.warpAffine(src,M,(w,h))
#     cv.imshow("result",dst)
# if __name__ == "__main__":
#     translate_image()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def arithmetic_demo():
#     src1=cv.imread("D:/picture1/lena2.jpg")
#     src2 = cv.imread("D:/picture1/lena1.jpg")
#     cv.imshow("input1", src1)
#     cv.imshow("input2", src2)
#     h, w = src1.shape[:2]
#     src2=cv.resize(src2,(w,h),interpolation=cv.INTER_LINEAR)
#     dst=cv.multiply(src1,src2)
#     cv.imshow("result",dst)
# if __name__ == "__main__":
#     arithmetic_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def logic_operater_demo():
#     src1=np.zeros((400,400,3),dtype=np.uint8)
#     src2 = np.zeros((400, 400, 3), dtype=np.uint8)
#     cv.rectangle(src1,(100,100),(300,300),(255,0,255),-1,cv.LINE_8)
#     cv.rectangle(src2, (20, 20), (220, 220), (255, 0, 0), -1, cv.LINE_8)
#     cv.imshow("input1", src1)
#     cv.imshow("input2", src2)
#     dst=cv.bitwise_xor(src1,src2)
#     cv.imshow("result",dst)
#     src3 = cv.imread("D:/picture1/lena2.jpg")
#     cv.imshow("input3", src3)
#     dst1=cv.bitwise_not(src3)
#     cv.imshow("not", dst1)
#     dst2 = cv.log(np.float32(src3))
#     dst3 = cv.sqrt(np.float32(src3))
#     cv.imshow("log", dst2)
#     cv.imshow("sqrt", dst3)
#
# if __name__ == "__main__":
#     logic_operater_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# def ligtness_contrast_demo():
#     src=cv.imread("D:/picture1/xueguan.tif")
#     cv.namedWindow("result",cv.WINDOW_AUTOSIZE)
#     cv.imshow("input",src)
#     empty=np.zeros(src.shape,src.dtype)
#     cv.createTrackbar("contrast", "result", 50, 100, do_nothing)
#     cv.createTrackbar("lightness", "result", 0, 200, do_nothing)
#     while(True):
#         cnt=cv.getTrackbarPos("contrast", "result")/50
#         beta = cv.getTrackbarPos("lightness", "result")
#         dst=cv.addWeighted(src,cnt,empty,0,beta)
#         cv.imshow("result",dst)
#         c=cv.waitKey(10)
#         if c==27:
#             break
# def do_nothing(pl):
#     print(pl)
# if __name__ == "__main__":
#     ligtness_contrast_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


#???????????????????????????????????????????????????????????????????????????????????????
# def do_nothing(pl):
#     print(pl)
# def lightness_saturation_demo():
#     a = 0
#     b = 0
#     src=cv.imread("D:/picture1/lena2.jpg")
#     cv.namedWindow("result",cv.WINDOW_AUTOSIZE)
#     cv.imshow("input",src)
#     HLS=cv.cvtColor(src,cv.COLOR_RGB2HLS)
#     hls=cv.split(HLS)
#     h,w=src.shape[:2]
#     cv.imshow("H",hls[0])
#     cv.imshow("L",hls[1])
#     cv.imshow("s",hls[2])
#     cv.createTrackbar("lightness", "result", 0,255, do_nothing)
#     cv.createTrackbar("saturation", "result", 0,255, do_nothing)
#     while(True):
#         tl=cv.getTrackbarPos("lightness", "result")
#         ts= cv.getTrackbarPos("saturation", "result")
#         if ((tl!=a)or(ts!=b)):
#             for row in range(h):
#                 for col in range(w):
#                     hls[2][row, col] += ts
#                     hls[1][row, col] += tl
#                     # if(hls[2][row, col]>255):
#                     #     hls[2][row, col]=255
#                     # if(hls[1][row, col]>255):
#                     #     hls[1][row, col]=255
#
#         a=tl
#         b=ts
#         dst = cv.merge(hls)
#         bgr = cv.cvtColor(dst, cv.COLOR_HLS2RGB)
#         bgr=np.clip(bgr,0,255)
#         cv.imshow("result", bgr)
#         c=cv.waitKey(10)
#         if c==27:
#             break
# lightness_saturation_demo()
# cv.waitKey(0)
# cv.destroyAllWindows()



# def color_space_demo():
#     src=cv.imread("D:/picture1/green1.jpg")
#     cv.namedWindow("input",cv.WINDOW_AUTOSIZE)
#     cv.imshow("input",src)
#     hsv=cv.cvtColor(src,cv.COLOR_BGR2HSV)
#     cv.imshow("HSV",hsv)
#     mask=cv.inRange(hsv,(35,43,46),(77,255,255))
#     cv.imshow("inrange",mask)
#     dst=np.zeros(src.shape,src.dtype)
#     dst= np.zeros(src.shape, src.dtype)
#     mask=cv.bitwise_not(mask)
#     cv.imshow("mask",mask)
#     result=cv.add(src,dst,mask=mask)
#     cv.imshow("result",result)
# if __name__ == "__main__":
#     color_space_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


#?????????????????????????????????????????????????????????????????????????????
# def float_image_demo():
#     src1 = cv.imread("D:/picture1/lena2.jpg")
#     src2= cv.imread("D:/picture1/t.tif")
#     h,w=src1.shape[:2]
#     src2=cv.resize(src2,(w,h),interpolation=cv.INTER_LINEAR)
#     f_src1=np.float32(src1)
#     f_src2 = np.float32(src2)
#     f_src=cv.divide(f_src1,1.1)
#     print(f_src)
#     cv.imshow("input",f_src)
#     dst=cv.convertScaleAbs(f_src)
#     cv.imshow("result1",dst)
#     cv.imshow("result2",f_src.astype(np.uint8))
#     cv.imshow("result3",np.uint8(f_src))
# if __name__ == "__main__":
#     float_image_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def statistics_demo():
#     src = cv.imread("D:/picture1/lena2.jpg")
#     h,w=src.shape[:2]
#     cv.imshow("input",src)
#     mbgr=cv.mean(src)
#     mbgr,devbgr=cv.meanStdDev(src)
#     print("blue mean:%d,green mean:%d,red mean:%d"%(mbgr[0],mbgr[1],mbgr[2]))
#     print("blue devbgr:%d,green devbgr:%d,red devbgr:%d" % (devbgr[0], devbgr[1], devbgr[2]))
#     gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#     bgr = cv.split(src)
#     t=cv.mean(gray)[0]
#     print(t)
#     cv.imshow("gray",gray)
#     cv.imshow("B",bgr[0])
#     cv.imshow("G", bgr[1])
#     cv.imshow("R", bgr[2])
#     hist=np.zeros([256],dtype=np.int32)
#     hist1 = np.zeros([256], dtype=np.int32)
#     hist2 = np.zeros([256], dtype=np.int32)
#     hist3= np.zeros([256], dtype=np.int32)
#     bin=np.zeros(gray.shape,gray.dtype)
#     for row in range(h):
#         for col in range(w):
#             pv=gray[row,col]
#             bv = bgr[0][row, col]
#             gv = bgr[1][row, col]
#             rv = bgr[2][row, col]
#             hist[pv]+=1
#             hist1[bv] += 1
#             hist2[gv] += 1
#             hist3[rv] += 1
#             if pv<t:
#                 bin[row,col]=0
#             else:
#                 bin[row,col]=255
#     print("min",np.min(gray))
#     print("max",np.max(gray))
#     cv.imshow("binary",bin)
#     plt.plot(hist,color="y")
#     plt.plot(hist1, color="b")
#     plt.plot(hist2, color="g")
#     plt.plot(hist3, color="r")
#     plt.xlim([0,256])
#     plt.ylim([0, 1500])
#     plt.show()
#
# if __name__ == "__main__":
#     statistics_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# def lookup_table_demo():
#     image=cv.imread("D:/picture1/lut_jet.png")
#     h,w=image.shape[:2]
#     lut=np.zeros((1,256,3),dtype=np.uint8)
#     for col in range(w):
#         lut[0,col]=image[h//2,col]
#     # lut[0,252]=lut[0,251]
#     # lut[0, 253] = lut[0, 251]
#     # lut[0, 254] = lut[0, 251]
#     # lut[0, 255] = lut[0, 251]
#     src=cv.imread("D:/picture1/xueguan.tif")
#     cv.imshow("input",src)
#     h,w=src.shape[:2]
#     for row in range(h):
#         for col in range(w):
#             b,r,g=src[row,col]
#             b=lut[0,b,0]
#             g = lut[0, g, 1]
#             r = lut[0, r, 2]
#             src[row,col]=(b,g,r)
#     cv.imshow("result",src)
# if __name__ == "__main__":
#     lookup_table_demo()
#     cv.waitKey(0)
#     cv.destroyAllWindows()

