import numpy as np
import cv2
import math

# 生成高斯矩阵
def getGaussMatrix(r):
    sum = 0
    sigma = r / 3
    GM = np.zeros((2 * r + 1, 2 * r + 1))

    for i in range (-r, r):
        for j in range (-r, r):
            GaussVal = (1 / (2 * math.pi * (sigma ** 2) )) * math.e ** ( (-(i ** 2 + j ** 2)) / 2 * (sigma ** 2) )
            GM[i + r, j + r] = GaussVal
            sum += GaussVal

    for i in range (-r, r):
        for j in range (-r, r):
            GM[i + r, j + r] /= sum

    return GM

def GaussianBlur(src, Matrix, width, height, r):

    dst = np.zeros( shape=(width, height),
                    dtype=src.dtype )

    for i in range (0, width):
        for j in range (0, height):
            mX = 0
            for x in range (i - r, i + r + 1):
                mY = 0
                for y in range (j - r, j + r + 1):
                    if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                        pass
                    else:
                        dst[i, j] += ( Matrix[mX, mY] * src[x, y] )
                    mY += 1
                mX += 1

    return dst

def addNoise(img):
    mean = 0    # 均值
    sigma = 10

    gauss = np.random.normal(mean, sigma, np.shape(img))
    dst = img + gauss

    dst[dst < 0] = 0
    dst[dst > 255] = 255

    return dst

def wienerFilter(img, h, K = 0.07):

    h /= np.sum(h)

    origin = np.copy(img)

    oriFFT = np.fft.fft2(origin)   # input傅立叶变换

    hFFT = np.fft.fft2(h, img.shape)  # 退化函数傅立叶变换

    WF = np.conj(hFFT) / (np.abs(hFFT) ** 2 + K)

    dst = oriFFT * WF

    dst = np.abs(np.fft.ifft2(dst))

    prop =  255 / dst.max()
    dst *= prop

    return dst

if __name__ == "__main__":
    img = cv2.imread('/Users/fever/Desktop/Study/数字图像处理/DIP_2019/DIP_Demo/matlab/lena.jpg', cv2.IMREAD_GRAYSCALE)  # 读图
    H_rows, W_cols = img.shape[:2]  # 长宽获取
    print(H_rows, W_cols)

    # 高斯模糊
    radius = 2
    Matrix = getGaussMatrix( radius )

    afterGB = GaussianBlur(img, Matrix, W_cols, H_rows, radius)

    # 高斯噪声
    afterGBN = addNoise(afterGB)
    afterGBN = afterGBN.astype(np.uint8)

    # 维纳滤波

    dst = wienerFilter(afterGBN, Matrix)
    dst = dst.astype(np.uint8)

    imgRes = np.hstack([img, afterGBN, dst])

    cv2.imshow("Result", imgRes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
