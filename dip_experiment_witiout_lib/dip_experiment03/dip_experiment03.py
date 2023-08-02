# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import random  # 用于产生高斯噪声

# 读取目标路径
image_filter = mpimg.imread('./noise.bmp')
image_noise = mpimg.imread('./Baboon.bmp')
image_edge = mpimg.imread('./contact_lens_original.bmp')
image_skeleton = mpimg.imread('./Fig0343(a)(skeleton_orig).bmp')


# 加入椒盐噪声
def add_salt_and_pepper_noise(image, SNR):
    """
    Add salt and pepper noise to a signal.

        Args:
            image (numpy.ndarray): The signal to add noise to.
            SNR: The probability of adding noise to each element of the signal.
    """
    # 初始化变量
    img = image
    img = img.copy()
    SNR = SNR
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 判断噪声是否合理
    if 0 <= SNR <= 1:
        print("这是一个合理的椒盐函数噪声范围")
    else:
        print("Error:这不是一个合理的椒盐函数噪声范围")
        exit()

    # 算法
    noise_num = int((1 - SNR) * img_width * img_height)

    for i in range(noise_num):
        rand_x = random.randint(0, img_height - 1)
        rand_y = random.randint(0, img_width - 1)

        if random.randint(0, 1) == 0:
            img[rand_x][rand_y] = 0
        else:
            img[rand_x][rand_y] = 255

    plt.subplot(451)
    plt.imshow(img, cmap='gray')
    plt.title("salt and pepper noise")


# 脉冲噪声
def add_pulse_noise(image, SNR, alpha):
    """
    Add pulse noise to a signal.

        Args:
            image (numpy.ndarray): The signal to add noise to.
            SNR: The probability of adding noise to each element of the signal.
            alpha (float): The amplitude of the noise.
    """
    # 初始化变量
    img = image
    img = img.copy()
    SNR = SNR
    alpha = alpha
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 判断噪声是否合理
    if 0 <= SNR <= 1:
        print("这是一个合理的冲激函数噪声范围")
    else:
        print("Error:这不是一个合理的冲激函数噪声范围")
        exit()

    # 算法
    noise_num = int((1 - SNR) * img_width * img_height)

    for i in range(noise_num):
        rand_x = random.randint(0, img_height - 1)
        rand_y = random.randint(0, img_width - 1)

        img[rand_x][rand_y] += int(img[rand_x][rand_y] * alpha)
        if img[rand_x][rand_y] < 0:
            img[rand_x][rand_y] = 0
        elif img[rand_x][rand_y] > 255:
            img[rand_x][rand_y] = 255

    plt.subplot(452)
    plt.imshow(img, cmap='gray')
    plt.title("pulse noise")


# 增加高斯噪音
def add_gaussian_noise(image, mean, var):
    """
    Add gaussian noise to a signal.

        Args:
            image(numpy.ndarray): The signal to add noise to.
            mean: The average value.
            var: The variance value.
    """
    # 初始化变量
    img = image
    img = img.copy()
    mean = mean
    var = var
    img_height = img.shape[0]
    img_width = img.shape[1]
    noise = np.random.normal(mean, var ** 0.5, img.shape)

    for j in range(img_height):
        for k in range(img_width):
            img[j][k] += noise[j][k]

            if img[j][k] < 0:
                img[j][k] = 0
            elif img[j][k] > 255:
                img[j][k] = 255

    plt.subplot(453)
    plt.imshow(img, cmap='gray')
    plt.title("gaussian noise")


# 均值滤波，24位
def mean_filter_24bits(image, size):
    """
    Mean filters used in the image.

        Args:
            image(numpy.ndarray): The original image.
            size(tuple): The size of the kernel.
    """
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

    plt.subplot(456)
    plt.imshow(img_right / 255, cmap='gray')
    plt.title("mean filter")


# 均值滤波，8位
def mean_filter_8bits(image, size):
    """
    Mean filters used in the image.

        Args:
            image(numpy.ndarray): The original image.
            size(tuple): The size of the kernel.
    """
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

    img_top = np.zeros([img_height+extend, img_width])
    img_bottom = np.zeros([img_height+2*extend, img_width])
    img_left = np.zeros([img_height+2*extend, img_width+extend])
    img_right = np.zeros([img_height+2*extend, img_width+2*extend])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])


    img_top[:, :] = np.insert(img[:, :], 0, row, axis=0)
    img_bottom[:, :] = np.insert(img_top[:, :], img_height + extend, row, axis=0)
    img_left[:, :] = np.insert(img_bottom[:, :], 0, col, axis=1)
    img_right[:, :] = np.insert(img_left[:, :], img_width + extend, col, axis=1)


    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right[j+a][k+b] * val

            img_right[j+extend][k+extend] = int(cnt)

            if img_right[j+extend][k+extend] < 0:
                img_right[j+extend][k+extend] = 0
            elif img_right[j+extend][k+extend] > 255:
                img_right[j+extend][k+extend] = 255

            cnt = 0

    return img_right


# 中值滤波
def mid_filter_24bits(image, size):
    """
    Mid filters were used in the image.

        Args:
            image(numpy.ndarray): The original image.
            size(tuple): The size of the kernel.
    """
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

    mid_list = []

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
    mid_num = int(size ** 2 / 2)

    for c in range(3):
        for j in range(range_h):
            for k in range(range_w):
                for a in range(size):
                    for b in range(size):
                        mid_list.append(img_right[j+a][k+b][c])

                mid_list.sort()
                img_right[j+extend][k+extend][c] = int(mid_list[mid_num])

                if img_right[j+extend][k+extend][c] < 0:
                    img_right[j+extend][k+extend][c] = 0
                elif img_right[j+extend][k+extend][c] > 255:
                    img_right[j+extend][k+extend][c] = 255

                mid_list = []

    plt.subplot(457)
    plt.imshow(img_right / 255, cmap='gray')
    plt.title("mid filter")


# 最大值滤波
def biggest_filter_24bits(image, size):
    """
    Biggest filters were used in the image.

        Args:
            image(numpy.ndarray): The original image.
            size(tuple): The size of the kernel.
    """
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

    biggest_list = []

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
    biggest_num = int(size ** 2 / 2)

    for c in range(3):
        for j in range(range_h):
            for k in range(range_w):
                for a in range(size):
                    for b in range(size):
                        biggest_list.append(img_right[j+a][k+b][c])

                biggest_list.sort()
                img_right[j+extend][k+extend][c] = int(biggest_list[biggest_num])

                if img_right[j+extend][k+extend][c] < 0:
                    img_right[j+extend][k+extend][c] = 0
                elif img_right[j+extend][k+extend][c] > 255:
                    img_right[j+extend][k+extend][c] = 255

                biggest_list = []

    plt.subplot(458)
    plt.imshow(img_right / 255, cmap='gray')
    plt.title("biggest filter")


# 拉普拉斯算子演示专用
def laplace_operator_show(image):
    """
        Laplace operator used in the image.

            Args:
                image(numpy.ndarray): The original image.
        """
    # 初始化变量
    img = image
    size = 3
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
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

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right[j + a][k + b] * kernel[a][b]

            img_right[j + extend][k + extend] -= int(cnt)

            if img_right[j + extend][k + extend] < 0:
                img_right[j + extend][k + extend] = 0
            elif img_right[j + extend][k + extend] > 255:
                img_right[j + extend][k + extend] = 255

            cnt = 0

    plt.subplot(4, 5, 11)
    plt.imshow(img_right / 255, cmap='gray')
    plt.title("laplace operator")


# 拉普拉斯算子传递专用
def laplace_operator(image):
    """
        Laplace operator used in the image.

            Args:
                image(numpy.ndarray): The original image.

            return:
                image_right(numpy.ndarray): The processed image.
        """
    # 初始化变量
    img = image
    size = 3
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
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

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right[j + a][k + b] * kernel[a][b]

            img_right[j + extend][k + extend] -= int(cnt)

            if img_right[j + extend][k + extend] < 0:
                img_right[j + extend][k + extend] = 0
            elif img_right[j + extend][k + extend] > 255:
                img_right[j + extend][k + extend] = 255

            cnt = 0

    return img_right


# Sobel算子演示专用
def sobel_operator_show(image):
    """
        Sobel operator used in the image.

            Args:
                image(numpy.ndarray): The original image.
        """
    # 初始化变量
    img = image
    size = 3
    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    extend = int(size / 2)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_top = np.zeros([img_height + extend, img_width])
    img_bottom = np.zeros([img_height + 2 * extend, img_width])
    img_left = np.zeros([img_height + 2 * extend, img_width + extend])
    img_right = np.zeros([img_height + 2 * extend, img_width + 2 * extend])
    img_right_copy = np.zeros([img_height + 2 * extend, img_width + 2 * extend])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])

    img_top[:, :] = np.insert(img[:, :], 0, row, axis=0)
    img_bottom[:, :] = np.insert(img_top[:, :], img_height + extend, row, axis=0)
    img_left[:, :] = np.insert(img_bottom[:, :], 0, col, axis=1)
    img_right[:, :] = np.insert(img_left[:, :], img_width + extend, col, axis=1)
    img_right_copy[:, :] = img_right[:, :]

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right[j + a][k + b] * kernel_horizontal[a][b]

            img_right[j + extend][k + extend] += int(cnt)

            if img_right[j + extend][k + extend] < 0:
                img_right[j + extend][k + extend] = 0
            elif img_right[j + extend][k + extend] > 255:
                img_right[j + extend][k + extend] = 255

            cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right_copy[j + a][k + b] * kernel_vertical[a][b]

            img_right_copy[j + extend][k + extend] += int(cnt)

            if img_right_copy[j + extend][k + extend] < 0:
                img_right_copy[j + extend][k + extend] = 0
            elif img_right_copy[j + extend][k + extend] > 255:
                img_right_copy[j + extend][k + extend] = 255

            cnt = 0

    img_right = np.divide(np.add(img_right, img_right_copy), 2)

    plt.subplot(4, 5, 12)
    plt.imshow(img_right / 255, cmap='gray')
    plt.title("sobel operator")


# Sobel算子传递专用
def sobel_operator(image):
    """
        Sobel operator used in the image.

            Args:
                image(numpy.ndarray): The original image.

            return:
                image_right(numpy.ndarray): The processed image.
        """
    # 初始化变量
    img = image
    size = 3
    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    extend = int(size / 2)

    img_height = img.shape[0]
    img_width = img.shape[1]

    img_top = np.zeros([img_height + extend, img_width])
    img_bottom = np.zeros([img_height + 2 * extend, img_width])
    img_left = np.zeros([img_height + 2 * extend, img_width + extend])
    img_right = np.zeros([img_height + 2 * extend, img_width + 2 * extend])
    img_right_copy = np.zeros([img_height + 2 * extend, img_width + 2 * extend])

    # 扩大矩阵
    row = np.zeros([extend, img_width])
    col = np.zeros([extend, img_height + 2 * extend])

    img_top[:, :] = np.insert(img[:, :], 0, row, axis=0)
    img_bottom[:, :] = np.insert(img_top[:, :], img_height + extend, row, axis=0)
    img_left[:, :] = np.insert(img_bottom[:, :], 0, col, axis=1)
    img_right[:, :] = np.insert(img_left[:, :], img_width + extend, col, axis=1)
    img_right_copy[:, :] = img_right[:, :]

    # 算法
    range_h = img_height + 2 * extend - size + 1
    range_w = img_width + 2 * extend - size + 1
    cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right[j + a][k + b] * kernel_horizontal[a][b]

            img_right[j + extend][k + extend] += int(cnt)

            if img_right[j + extend][k + extend] < 0:
                img_right[j + extend][k + extend] = 0
            elif img_right[j + extend][k + extend] > 255:
                img_right[j + extend][k + extend] = 255

            cnt = 0

    for j in range(range_h):
        for k in range(range_w):
            for a in range(size):
                for b in range(size):
                    cnt += img_right_copy[j + a][k + b] * kernel_vertical[a][b]

            img_right_copy[j + extend][k + extend] += int(cnt)

            if img_right_copy[j + extend][k + extend] < 0:
                img_right_copy[j + extend][k + extend] = 0
            elif img_right_copy[j + extend][k + extend] > 255:
                img_right_copy[j + extend][k + extend] = 255

            cnt = 0

    img_right = np.divide(np.add(img_right, img_right_copy), 2)

    return img_right


# 扩大矩阵
def extend_matrix(image, size):
    img = image
    if size[0] == size[1] and size[0] % 2 == 1:
        size = size[0]
    else:
        print("错误的核大小！")
        exit()
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

    return img_right


# 幂律变换
def power_trans(image, c, gamma):
    c = c
    gamma = gamma
    img = image / 255

    img_width = img.shape[1]
    img_height = img.shape[0]

    for i in range(img_height):
        for j in range(img_width):
            img[i][j] = c * img[i][j] ** gamma

    return img


# 医学图像处理
def med_img_pro(image):
    img_ori = extend_matrix(image, (3, 3))
    img_laplace = laplace_operator(image)
    img_add1 = np.add(img_ori, img_laplace)
    img_sobel = sobel_operator(img_add1)
    img_mean_filter = mean_filter_8bits(img_sobel, (3, 3))
    img_add1 = extend_matrix(img_add1, (5, 5))
    img_mul = np.multiply(img_add1, img_mean_filter)
    img_ori = extend_matrix(img_ori, (5, 5))
    img_add2 = np.add(img_ori, img_mul)
    img_final = power_trans(img_add2, 1, 0.5)
    plt.subplot(4, 5, 13)
    plt.imshow(img_final, cmap='gray')
    plt.title("final version")


# 主函数
if __name__ == '__main__':
    add_salt_and_pepper_noise(image_noise, 0.6)
    add_pulse_noise(image_noise, 0.6, 1.1)
    add_gaussian_noise(image_noise, 1, 1)
    mean_filter_24bits(image_filter, (5, 5))
    mid_filter_24bits(image_filter, (3, 3))
    biggest_filter_24bits(image_filter, (3, 3))
    laplace_operator_show(image_edge)
    sobel_operator_show(image_edge)
    med_img_pro(image_skeleton)
    plt.tight_layout()
    plt.show()