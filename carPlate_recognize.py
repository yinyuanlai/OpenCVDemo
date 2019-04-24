'''
车牌基于蓝色区域进行定位分割
author:yinyuanlai
date:2019-4-24
'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

####################################################################
'''定位部分'''
car = cv.imread('G://images//car.png', 1)
cv.imshow('car', car)
'''蓝色和黄色所对应的色彩空间'''
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
lower_yellow = np.array([15, 55, 55])
upper_yellow = np.array([50, 255, 255])
hsv = cv.cvtColor(car, cv.COLOR_BGR2HSV)  # 将BGR图像转化到HSV的颜色空间
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
mask_plate = cv.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow)
# cv.imshow('mask_plate',mask_plate)
# 根据阈值找到对应颜色
mask = cv.cvtColor(mask_plate, cv.COLOR_BGR2GRAY)
Matrix = np.ones((20, 20), np.uint8)
mask1 = cv.morphologyEx(mask, cv.MORPH_CLOSE, Matrix)
mask = cv.morphologyEx(mask1, cv.MORPH_OPEN, Matrix)  # 形态学开运算
ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)  # 二值化进而获取轮廓
# cv.imshow('mask',mask)
_, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓 contours
# 寻找轮廓最大的 定位车牌
area_list = []  # 定义新列表存储车牌的面积
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    area_list.append(area)
    if area > 3000:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
plate = cv.drawContours(car.copy(), [box], -1, (0, 255, 0), 3)
cv.imshow('plate', plate)
####################################################################################################
'''ROI裁剪'''
ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
ys_sorted_index = np.argsort(ys)
xs_sorted_index = np.argsort(xs)
x1 = box[xs_sorted_index[0], 0]
x2 = box[xs_sorted_index[3], 0]
y1 = box[ys_sorted_index[0], 1]
y2 = box[ys_sorted_index[3], 1]
ROI_plate = plate[y1:y2, x1:x2]
cv.imshow('ROI_plate', ROI_plate)
####################################################################################################
'''ROI字符分割'''
ROI_plate_gray = cv.cvtColor(ROI_plate, cv.COLOR_BGR2GRAY)  # 灰度化
ROI_plate_blur = cv.GaussianBlur(ROI_plate_gray, (5, 5), 0)  # 高斯滤波
ret, ROI_plate_Binary = cv.threshold(ROI_plate_blur, 127, 255, cv.THRESH_BINARY)  # 二值化
# 形态学腐蚀 去除边框
kernel = np.ones((5, 5), dtype=np.uint8)
ROI_erode = cv.erode(ROI_plate_Binary, kernel, iterations=1)
cv.imshow('ROI_erode', ROI_erode)
# 根据宽度 裁剪7个字符
width = ROI_erode.shape[1]
height = ROI_erode.shape[0]
plt.subplot(241), plt.imshow(ROI_erode[0:height, 0:np.uint8(height / 2)], cmap='gray')  # 裁剪出了鲁
plt.subplot(242), plt.imshow(ROI_erode[0:height, np.uint8(height / 2):height], cmap='gray')  # 裁剪出C
# 算出后面5个 每个字符的间距
size = np.uint8((width - height) / 5)
word_0 = ROI_erode[0:height, height + 0 * size:height + 1 * size]  # 第一个字符
word_1 = ROI_erode[0:height, height + 1 * size:height + 2 * size]  # 第二个字符
word_2 = ROI_erode[0:height, height + 2 * size:height + 3 * size]  # 第三个字符
word_3 = ROI_erode[0:height, height + 3 * size:height + 4 * size]  # 第四个字符
word_4 = ROI_erode[0:height, height + 4 * size:height + 5 * size]  # 第五个字符
plt.subplot(243), plt.imshow(word_0, cmap='gray')
plt.subplot(244), plt.imshow(word_1, cmap='gray')
plt.subplot(245), plt.imshow(word_2, cmap='gray')
plt.subplot(246), plt.imshow(word_3, cmap='gray')
plt.subplot(247), plt.imshow(word_4, cmap='gray')
plt.show()
cv.waitKey(0)
