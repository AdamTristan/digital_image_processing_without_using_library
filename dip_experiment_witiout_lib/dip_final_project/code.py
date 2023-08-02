# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np


# 读取目标路径
image_apple = mpimg.imread('./red_apple.bmp')


# 显示原图像
def original_display(image1, image2, pixel):
    img1 = image1
    img2 = image2
    pixel_width = pixel
    img_copy = np.copy(img1)
    img_height = img2.shape[0]
    img_width = img2.shape[1]
    y_top = img_height
    y_bottom = 0
    x_left = img_width
    x_right = 0

    for i in range(img_height - 10):
        for j in range(img_width - 10):
            if img2[i, j] == 1 and i < y_top:
                y_top = i
            elif img2[i, j] == 1 and i > y_bottom:
                y_bottom = i
            if img2[i, j] == 1 and j < x_left:
                x_left = j
            elif img2[i, j] == 1 and j > x_right:
                x_right = j

    for i in range(x_left, x_right):
        for j in range(y_top-pixel_width, y_top+pixel_width):
            img_copy[j, i, :] = 0
    for i in range(x_left, x_right):
        for j in range(y_bottom-pixel_width, y_bottom+pixel_width):
            img_copy[j, i, :] = 0
    for i in range(y_top, y_bottom):
        for j in range(x_left-pixel_width, x_left+pixel_width):
            img_copy[i, j, :] = 0
    for i in range(y_top, y_bottom):
        for j in range(x_right-pixel_width, x_right+pixel_width):
            img_copy[i, j, :] = 0

    plt.subplot(321)
    plt.imshow(img_copy / 255)
    plt.title("original")


# 均值滤波，24位(R, G, B)
def mean_filter_24bits(image, size):
    # 初始化变量
    img = image
    if size[0] == size[1] and size[0] % 2 == 1:
        size = size[0]
    else:
        print("错误的核大小！")
        exit()

    val = 1 / size ** 2
    kernel = np.full((size, size), val)
    extend = int(size / 2)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_top = np.zeros([img_height+extend, img_width, 3])
    img_bottom = np.zeros([img_height+2*extend, img_width, 3])
    img_left = np.zeros([img_height+2*extend, img_width+extend, 3])
    img_right = np.zeros([img_height+2*extend, img_width+2*extend, 3])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])

    for c in range(3):
        img_top[:, :, c] = np.insert(img[:, :, c], 0, row, axis=0)
        img_bottom[:, :, c] = np.insert(img_top[:, :, c], img_height + extend, row, axis=0)
        img_left[:, :, c] = np.insert(img_bottom[:, :, c], 0, col, axis=1)
        img_right[:, :, c] = np.insert(img_left[:, :, c], img_width + extend, col, axis=1)

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for c in range(3):
        for j in range(range_h):
            for k in range(range_w):
                for a in range(size):
                    for b in range(size):
                        cnt += img_right[j+a][k+b][c] * val

                img_right[j+extend][k+extend][c] = int(cnt)

                if img_right[j+extend][k+extend][c] < 0:
                    img_right[j+extend][k+extend][c] = 0
                elif img_right[j+extend][k+extend][c] > 255:
                    img_right[j+extend][k+extend][c] = 255

                cnt = 0

    plt.subplot(322)
    plt.imshow(img_right / 255)
    plt.title("mean filter")

    return img_right


# RGB变HSV
def RGB2HSV(image):
    img = image

    R = img[:, :, 0] / 255
    G = img[:, :, 1] / 255
    B = img[:, :, 2] / 255
    img_height = img.shape[0]
    img_width = img.shape[1]
    max_matrix = np.zeros((img_height, img_width))
    min_matrix = np.zeros((img_height, img_width))
    H = np.zeros((img_height, img_width))
    S = np.zeros((img_height, img_width))
    V = np.zeros((img_height, img_width))
    delta = np.zeros((img_height, img_width))

    for i in range(img_height):
        for j in range(img_width):
            max_matrix[i, j] = max(R[i, j], G[i, j], B[i, j])
            min_matrix[i, j] = min(R[i, j], G[i, j], B[i, j])
            delta[i, j] = max_matrix[i, j] - min_matrix[i, j]

    # 计算V
    V = max_matrix

    # 计算S
    for i in range(img_height):
        for j in range(img_width):
            if max_matrix[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = delta[i, j] / max_matrix[i, j]

    # 计算H
    for i in range(img_height):
        for j in range(img_width):
            if max_matrix[i, j] == R[i, j]:
                H[i, j] = (0 + (G[i, j] - B[i, j]) / delta[i, j]) / 6
            elif max_matrix[i, j] == G[i, j]:
                H[i, j] = (2 + (G[i, j] - B[i, j]) / delta[i, j]) / 6
            elif max_matrix[i, j] == B[i, j]:
                H[i, j] = (4 + (G[i, j] - B[i, j]) / delta[i, j]) / 6

            if H[i, j] < 0:
                H[i, j] += 360

    img[:, :, 0] = H
    img[:, :, 1] = S
    img[:, :, 2] = V

    plt.subplot(323)
    plt.imshow(img)
    plt.title("HSV")

    return img


# 阈值分割
def threshold(image):
    lower_red = np.array([20, 20, 70]) / 255
    upper_red = np.array([190, 255, 255]) / 255
    img = image
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_binary = np.zeros((img_height, img_width))

    for i in range(img_height):
        for j in range(img_width):
            if lower_red[0] <= img[i, j, 0] <= upper_red[0] and lower_red[1] <= img[i, j, 1] <= upper_red[1] and lower_red[2] <= img[i, j, 2] <= upper_red[2]:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1

    plt.subplot(324)
    plt.imshow(img_binary, cmap='gray')
    plt.title("binary")

    return img_binary


# 腐蚀
def erosion(image, type, times):
    img = image
    isErosion = 0

    size = (3, 3)

    if size[0] == size[1] and size[0] % 2 == 1:
        size = size[0]
    else:
        print("错误的核大小！")
        exit()

    val = 1 / size ** 2
    kernel = np.full((size, size), val)
    extend = int(size / 2)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_top = np.zeros([img_height + extend, img_width])
    img_bottom = np.zeros([img_height + 2 * extend, img_width])
    img_left = np.zeros([img_height + 2 * extend, img_width + extend])
    img_right = np.zeros([img_height + 2 * extend, img_width + 2 * extend])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])

    img_top[:, :] = np.insert(img[:, :], 0, row, axis=0)
    img_bottom[:, :] = np.insert(img_top[:, :], img_height + extend, row, axis=0)
    img_left[:, :] = np.insert(img_bottom[:, :], 0, col, axis=1)
    img_right[:, :] = np.insert(img_left[:, :], img_width + extend, col, axis=1)
    img_copy = np.copy(img_right)

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1

    if type == 'full':
        for i in range(times):
            for j in range(range_h):
                for k in range(range_w):
                    for a in range(size):
                        if isErosion == 1:
                            break
                        for b in range(size):
                            if img_copy[j+a-1][k+b-1] == 0:
                                isErosion = 1
                                img_right[j, k] = 0
                                break
                    isErosion = 0
            img_copy = np.copy(img_right)

    if type == 'cross':
        for i in range(times):
            for j in range(range_h):
                for k in range(range_w):
                    if img_copy[j-1][k] == 1 and img_copy[j][k] == 1 and img_copy[j][k-1] == 1 and img_copy[j+1][k] == 1 and img_copy[j][k+1] == 1:
                        img_right[j, k] = 1
                    else:
                        img_right[j, k] = 0
            img_copy = np.copy(img_right)

    plt.subplot(325)
    plt.imshow(img_right, cmap='gray')
    plt.title("erosion")

    return img_right


# 膨胀
def expansion(image, type, times):
    img = image
    isErosion = 0

    size = (3, 3)

    if size[0] == size[1] and size[0] % 2 == 1:
        size = size[0]
    else:
        print("错误的核大小！")
        exit()

    val = 1 / size ** 2
    kernel = np.full((size, size), val)
    extend = int(size / 2)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_top = np.zeros([img_height + extend, img_width])
    img_bottom = np.zeros([img_height + 2 * extend, img_width])
    img_left = np.zeros([img_height + 2 * extend, img_width + extend])
    img_right = np.zeros([img_height + 2 * extend, img_width + 2 * extend])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])

    img_top[:, :] = np.insert(img[:, :], 0, row, axis=0)
    img_bottom[:, :] = np.insert(img_top[:, :], img_height + extend, row, axis=0)
    img_left[:, :] = np.insert(img_bottom[:, :], 0, col, axis=1)
    img_right[:, :] = np.insert(img_left[:, :], img_width + extend, col, axis=1)
    img_copy = np.copy(img_right)

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1

    if type == 'full':
        for i in range(times):
            for j in range(range_h):
                for k in range(range_w):
                    for a in range(size):
                        if isErosion == 1:
                            break
                        for b in range(size):
                            if img_copy[j+a-1][k+b-1] == 0:
                                isErosion = 1
                                img_right[j, k] = 0
                                break
                    isErosion = 0
            img_copy = np.copy(img_right)

    if type == 'cross':
        for i in range(times):
            for j in range(range_h):
                for k in range(range_w):
                    if img_copy[j-1][k] == 1 or img_copy[j][k] == 1 or img_copy[j][k-1] == 1 or img_copy[j+1][k] == 1 or img_copy[j][k+1] == 1:
                        img_right[j, k] = 1
                    else:
                        img_right[j, k] = 0
            img_copy = np.copy(img_right)

    plt.subplot(326)
    plt.imshow(img_right, cmap='gray')
    plt.title("opening")

    return img_right


# 主函数
if __name__ == '__main__':
    img_filter = mean_filter_24bits(image_apple, (3, 3))
    img_HSV = RGB2HSV(img_filter)
    img_binary = threshold(img_HSV)
    img_erosion = erosion(img_binary, 'cross', 5)
    img_opening = expansion(img_erosion, 'cross', 3)
    original_display(image_apple, img_opening, 2)

    plt.tight_layout()
    plt.show()