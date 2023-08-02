import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

# 读取目标路径
image_to = mpimg.imread('./tungsten_original.bmp')
image_esn = mpimg.imread('./embedded_square_noisy.bmp')
image_cherry_dark = mpimg.imread('./cherry(dark).bmp')
image_cherry_light = mpimg.imread('./cherry(light).bmp')

# 直方图显示
def histogram(image):
    img = image
    width = img.shape[1]
    height = img.shape[0]
    data = np.zeros(width*height)

    for i in range(height):
        for j in range(width):
            data[i * width + j] = img[i][j]

    plt.subplot(321)
    plt.hist(data, bins=255, range=(0, 255))
    plt.title("histogram")
    plt.xlabel("intensity")
    plt.ylabel("num")

# 灰度线性变换
def grayscale_trans(image, k, b):
    img = image
    width = img.shape[1]
    height = img.shape[0]
    origin_mat = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            origin_mat[i][j] = k * img[i][j] + b
            if origin_mat[i][j] > 255:
                origin_mat[i][j] = 255
            elif origin_mat[i][j] < 0:
                origin_mat[i][j] = 0

    plt.subplot(322)
    plt.imshow(origin_mat, cmap='gray', vmin=0, vmax=255)
    plt.title("grayscale transform")

# 直方图均衡
def hist_equal(image):
    img = image
    width = img.shape[1]
    height = img.shape[0]
    gray_num = np.zeros(256)
    gray_num_copy = np.zeros(256)
    gray_num_trans = np.zeros(256)
    pixel_num = width * height
    all_diff_num = []
    x = list(range(256))

    # 寻找各个灰度值的个数
    for i in range(height):
        for j in range(width):
            gray_intensity = img[i][j]
            gray_num[gray_intensity] += 1
            gray_num_copy[gray_intensity] = gray_num[gray_intensity]

    # 计算原始图像的灰度分布频率
    for k in range(256):
        gray_num[k] = round(gray_num[k] / pixel_num, 20)

    # 计算原始图像的灰度累积分布频率
    for m in range(255, -1, -1):
        for n in range(m):
            gray_num[m] += gray_num[n]

    # 归一化并四舍五入
    for k in range(256):
        gray_num[k] = int(round(gray_num[k] * 255, 0))

    # 寻找所有不重复的索引
    for k in gray_num:
        if k not in all_diff_num:
            all_diff_num.append(int(k))

    # 获得新的柱状图矩阵
    for i in all_diff_num:
        for j in range(256):
            if i == gray_num[j]:
                gray_num_trans[i] += gray_num_copy[j]

    plt.subplot(323)
    plt.title("histogram equalization")
    plt.xlabel("intensity")
    plt.ylabel("num")
    plt.bar(x, gray_num_trans)

# 线性图像增强
def image_argument(image, k, b):
    img = image
    width = img.shape[1]
    height = img.shape[0]
    origin_mat = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            origin_mat[i][j] = k * img[i][j] + b
            if origin_mat[i][j] > 255:
                origin_mat[i][j] = 255
            elif origin_mat[i][j] < 0:
                origin_mat[i][j] = 0

    plt.subplot(324)
    plt.imshow(origin_mat, cmap='gray', vmin=0, vmax=255)
    plt.title("image argument")

# 直方图规格化
def hist_spe(src_image, obj_image):

    # 原图像数据收集
    src_img = image_cherry_dark
    src_width = src_img.shape[1]
    src_height = src_img.shape[0]
    src_pixel_num = src_height * src_width

    # 目标图像数据收集
    obj_img = image_cherry_light
    obj_width = obj_img.shape[1]
    obj_height = obj_img.shape[0]
    obj_pixel_num = obj_height * obj_width

    # 建立数组和索引
    src_gray_num = np.zeros([3, 256])
    obj_gray_num = np.zeros([3, 256])
    map_all_diff_num_1 = []
    index_list1 = [0]
    index = 0
    map_all_diff_num_2 = []
    index_list2 = [0]
    map_all_diff_num_3 = []
    index_list3 = [0]
    min_index = np.zeros([3, 256])

    # 创建新图像数组
    new_img = np.zeros([src_height, src_width, 3])

    # 寻找原始灰度值的个数
    for k in range(3):
        for i in range(src_height):
            for j in range(src_width):
                src_gray_intensity = src_img[i][j][k]
                src_gray_num[k][src_gray_intensity] += 1
                # gray_num_copy[gray_intensity] = gray_num[gray_intensity]

    # 计算原始图像的灰度分布频率
    for i in range(3):
        for j in range(256):
            src_gray_num[i][j] = round(src_gray_num[i][j] / src_pixel_num, 20)

    # 计算原始图像的灰度累积分布频率
    for k in range(3):
        for m in range(255, -1, -1):
            for n in range(m):
                src_gray_num[k][m] += src_gray_num[k][n]

    # 寻找目标灰度值的个数
    for k in range(3):
        for i in range(obj_height):
            for j in range(obj_width):
                obj_gray_intensity = obj_img[i][j][k]
                obj_gray_num[k][obj_gray_intensity] += 1
                # gray_num_copy[gray_intensity] = gray_num[gray_intensity]

    # 计算目标图像的灰度分布频率
    for i in range(3):
        for j in range(256):
            obj_gray_num[i][j] = round(obj_gray_num[i][j] / obj_pixel_num, 20)

    # 计算目标图像的灰度累积分布频率
    for k in range(3):
        for m in range(255, -1, -1):
            for n in range(m):
                obj_gray_num[k][m] += obj_gray_num[k][n]

    # 原图像保留两位
    for k in range(3):
        for m in range(256):
            src_gray_num[k][m] = round(src_gray_num[k][m], 2)

    # 目标图像保留两位
    for k in range(3):
        for m in range(256):
            obj_gray_num[k][m] = round(obj_gray_num[k][m], 2)

    for i in range(3):
        for j in range(256):
            min_num = 1
            for k in range(256):
                if abs(src_gray_num[i][j] - obj_gray_num[i][k]) < min_num:
                    min_num = abs(src_gray_num[i][j] - obj_gray_num[i][k])
                    min_index[i][j] = k

    # 分离数组
    map_num_1 = min_index[0][:]
    map_num_2 = min_index[1][:]
    map_num_3 = min_index[2][:]

    # 寻找原图像所有不重复的索引以及最后一次出现索引的下标
    for k in range(256):
        if map_num_1[k] not in map_all_diff_num_1:
            map_all_diff_num_1.append(int(map_num_1[k]))
        if map_num_1[k] != map_num_1[index]:
            index_list1.append(int(k))
            index = k
    index_list1.append(int(256))
    index = 0
    for k in range(256):
        if map_num_2[k] not in map_all_diff_num_2:
            map_all_diff_num_2.append(int(map_num_2[k]))
        if map_num_2[k] != map_num_2[index]:
            index_list2.append(int(k))
            index = k
    index_list2.append(int(256))
    index = 0
    for k in range(256):
        if map_num_3[k] not in map_all_diff_num_3:
            map_all_diff_num_3.append(int(map_num_3[k]))
        if map_num_3[k] != map_num_3[index]:
            index_list3.append(int(k))
            index = k
    index_list3.append(int(256))

    # 存放映射数据至新矩阵
    for j in range(src_height):
        for k in range(src_width):
            for i in range(len(map_all_diff_num_1)):
                if index_list1[i] <= src_img[j][k][0] < index_list1[i + 1]:
                    new_img[j][k][0] = map_all_diff_num_1[i]
                    break

    for j in range(src_height):
        for k in range(src_width):
            for i in range(len(map_all_diff_num_2)):
                if index_list2[i] <= src_img[j][k][1] < index_list2[i + 1]:
                    new_img[j][k][1] = map_all_diff_num_2[i]
                    break

    for j in range(src_height):
        for k in range(src_width):
            for i in range(len(map_all_diff_num_3)):
                if index_list3[i] <= src_img[j][k][2] < index_list3[i + 1]:
                    new_img[j][k][2] = map_all_diff_num_3[i]
                    break

    # 显示图像
    plt.subplot(325)
    plt.imshow(new_img / 255)
    plt.title("histogram specification")


if __name__ == '__main__':
    histogram(image_to)
    grayscale_trans(image_to, 1.2, 1)
    hist_equal(image_esn)
    image_argument(image_esn, 2, 1)
    hist_spe(image_cherry_dark, image_cherry_light)
    plt.tight_layout()
    plt.show()