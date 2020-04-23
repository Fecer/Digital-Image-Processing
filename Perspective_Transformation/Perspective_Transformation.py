import numpy as np
import cv2

# 获取变幻矩阵
def perspectiveTransform(origin, object):
    if origin.shape[0] != object.shape[0]:  # 坐标点数量不相同
        print("Different point quantity")
    if origin.shape[0] < 4:  # 坐标点数量过少
        print("Deficient origin point quantity")

    pointNum = origin.shape[0]  # 坐标点数量

    # 四个点拼在一起
    # A( 8 * 8 )  B( 8 * 1 )
    # A * warpMatrix = B
    A = np.zeros((pointNum * 2, 8))
    B = np.zeros((pointNum * 2, 1))

    # 赋值
    for i in range(0, pointNum):
        # 提取第i个初始点和目标点
        Ai = origin[i, :]
        Bi = object[i, :]

        # 偶数行
        A[i * 2, :] = [Ai[0], Ai[1], 1, 0, 0, 0, -Ai[0] * Bi[0], -Ai[1] * Bi[0]]
        # 奇数行
        A[i * 2 + 1, :] = [0, 0, 0, Ai[0], Ai[1], 1, -Ai[0] * Bi[1], -Ai[1] * Bi[1]]

        B[i * 2] = Bi[0]
        B[i * 2 + 1] = Bi[1]

    A = np.mat(A)  # 将A转换成矩阵

    warpMatrix = A.I * B  # A逆 * B求得warpMatrix

    warpMatrix = np.array(warpMatrix).T[0]  # 转置
    warpMatrix = np.insert(arr=warpMatrix,  # 插入a33 = 1
                           obj=warpMatrix.shape[0],
                           values=1.0,
                           axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))  # 转化为3*3矩阵

    return warpMatrix

# 按矩阵变换
def perspectiveWarp(img, Matrix, H_rows, W_cols):
    # 初始化结果图
    dstImg = np.zeros((W_cols, H_rows, img.shape[2]), dtype=img.dtype)
    # 变换矩阵求逆
    Imatrix = np.linalg.inv(Matrix)
    # 依照逆变幻矩阵 求映射
    for i in range(0, H_rows):
        for j in range(0, W_cols):
            srcX = int((Imatrix[0, 0] * j + Imatrix[0, 1] * i + Imatrix[0, 2]) / (Imatrix[2, 0] * j + Imatrix[2, 1] * i + 1))
            srcY = int((Imatrix[1, 0] * j + Imatrix[1, 1] * i + Imatrix[1, 2]) / (Imatrix[2, 0] * j + Imatrix[2, 1] * i + 1))
            if (0 <= srcX < (H_rows - 1) and 0 <= srcY < (W_cols - 1)):
                dstImg[i, j] = img[srcY, srcX]

    return dstImg

if __name__ == "__main__":
    img = cv2.imread('/Users/fever/Desktop/Study/数字图像处理/DIP_2019/DIP_Demo/matlab/lena.jpg') # 读图
    H_rows, W_cols = img.shape[:2]  # 长宽获取
    print(H_rows, W_cols)

    pts1 = np.float32([[0, 0], [W_cols, 0], [0, H_rows], [H_rows, W_cols]])

    # pts2 = np.float32([[0, 20], [400, 30], [20, 400], [300, 300]])
    print("Format:x,y")
    x0, y0 = eval(input("Upleft:"))
    x1, y1 = eval(input("Upright:"))
    x2, y2 = eval(input("Downleft:"))
    x3, y3 = eval(input("Downright:"))

    # 确保图像不扭曲
    assert x1 > x0
    assert x3 > x2
    assert y2 > y0
    assert y3 > y1

    pts2 = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    Matrix = perspectiveTransform(pts1, pts2)  # 得到变幻矩阵

    dst = perspectiveWarp(img, Matrix, H_rows, W_cols) # 透视变换

    imgRes = np.hstack([img, dst])
    cv2.imshow("Result", imgRes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
