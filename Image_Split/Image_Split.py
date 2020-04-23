import random
import numpy as np
from matplotlib import pyplot as plt
import cv2

# 计算当前与上一次聚类中的差异
def lossFunction(curCenter, preCenter):

    curCenter = np.array(curCenter)
    preCenter = np.array(preCenter)

    res = np.sum((curCenter - preCenter) ** 2)

    return res

# 分割
def splitImg(img, c):

    pixlDist = [] # 像素和聚类中心的距离
    H_rows, W_cols = img.shape[:2] # 形状

    labels = np.zeros((H_rows, W_cols)) # 标签

    for i in range (0, H_rows):
        for j in range (0, W_cols):

            for k in range (0, len(c)):
                dist = np.sum( abs((img[i, j]).astype(int) - c[k].astype(int)) ** 2) # 差平方
                pixlDist.append(dist)

            labels[i, j] = int(pixlDist.index(min(pixlDist))) # 确定标签，以最近的中心为准
            pixlDist = [] # 初始化

    return labels

def kMeans (img, cnt, limit, H_rows, W_cols):

    origin = np.copy(img)
    labels = np.zeros((H_rows, W_cols))
    curCenter = []
    j = 0
    loss = 9999999

    # 随机行、列中心
    firstCenterCol = [i for i in range(0, W_cols)]
    random.shuffle(firstCenterCol)
    firstCenterCol = firstCenterCol[:cnt]
    firstCenterRow = [i for i in range (0, H_rows)]
    random.shuffle(firstCenterRow)
    firstCenterRow = firstCenterRow[:cnt]

    #当前中心
    for i in range (0, cnt):
        curCenter.append(origin[firstCenterRow[i], firstCenterCol[i]])
    labels = splitImg(origin, curCenter)
    
    while 1:
        if loss <= limit:
            break
        preCenter = curCenter.copy()    # 保存上一次
        for k in range (0, cnt):
            cur = np.where(labels == k)
            curCenter[k] = sum(origin[cur].astype(int)) / len(origin[cur])
            
        labels = splitImg(origin, curCenter)

        loss = lossFunction(curCenter, preCenter)
        j += 1
        print("Loss:" + str(loss))

    return labels

if __name__ == "__main__":
    limit = 1
    k = 3

    img = cv2.imread('/Users/fever/Desktop/Study/数字图像处理/DIP_2019/DIP_Demo/matlab/5.jpeg', cv2.IMREAD_COLOR)  # 读图
    H_rows, W_cols = img.shape[:2]  # 长宽获取
    print(H_rows, W_cols)
    b, g, r = cv2.split(img)
    origin = cv2.merge([r, g, b])

    labels = kMeans(img=img, cnt=k, limit=limit, H_rows=H_rows, W_cols=W_cols)
    plt.subplot(1, 2, 1)
    plt.title("Origin Image")
    plt.imshow(origin)

    plt.subplot(1, 2, 2)
    plt.title("K-means")
    plt.imshow(labels / 3, "gray")
    plt.show()
