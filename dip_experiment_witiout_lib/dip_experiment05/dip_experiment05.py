# -*- coding: utf-8 -*-
import heapq
from collections import defaultdict
from PIL import Image


# 定义Huffman编码树节点类
class HuffmanNode:
    def __init__(self, pixel=None, freq=0):
        self.pixel = pixel  # 图像像素值
        self.freq = freq  # 频率
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# 生成Huffman编码树
def build_huffman_tree(image):
    pixel_freq = defaultdict(int)
    width, height = image.size

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            pixel_freq[pixel] += 1

    nodes = [HuffmanNode(pixel, freq) for pixel, freq in pixel_freq.items()]
    heapq.heapify(nodes)

    while len(nodes) > 1:
        left_node = heapq.heappop(nodes)
        right_node = heapq.heappop(nodes)
        parent_node = HuffmanNode(freq=left_node.freq + right_node.freq)
        parent_node.left = left_node
        parent_node.right = right_node
        heapq.heappush(nodes, parent_node)

    return nodes[0]


# 生成Huffman编码表
def build_huffman_table(node, prefix="", huffman_table={}):
    if node.pixel is not None:
        huffman_table[node.pixel] = prefix
    else:
        build_huffman_table(node.left, prefix + "0", huffman_table)
        build_huffman_table(node.right, prefix + "1", huffman_table)

    return huffman_table


# 将图像转换为Huffman编码
def compress_image(image, huffman_table):
    width, height = image.size
    compressed_bits = ""

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            compressed_bits += huffman_table[pixel]

    return compressed_bits


# 计算压缩率
def calculate_compression_ratio(image, compressed_bits):
    original_size = image.width * image.height * 8
    compressed_size = len(compressed_bits)
    compression_ratio = original_size / compressed_size
    return compression_ratio


def huffman(image_path):
    # 加载BMP图像
    image_path = image_path
    image = Image.open(image_path)

    # 生成Huffman编码树和编码表
    huffman_tree = build_huffman_tree(image)
    huffman_table = build_huffman_table(huffman_tree)

    # 压缩图像并计算压缩率
    compressed_bits = compress_image(image, huffman_table)
    compression_ratio = calculate_compression_ratio(image, compressed_bits)

    print(image_path + " Huffman Compression Ratio: {:.2f}".format(compression_ratio))


# 计算压缩率
def calculate_compression_ratio_2(image, compressed_data):
    original_size = image.width * image.height * 8
    compressed_size = len(compressed_data) * 12  # 每个编码占用12比特
    compression_ratio = original_size / compressed_size
    return compression_ratio


# 将图像数据转换为八比特序列
def convert_image_to_data(image):
    image = image.convert("P")
    width, height = image.size
    data = []

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            data.append(pixel)

    return data


# 将图像数据转换为八比特序列
def convert_image_to_data_2(image, i):
    image = image.convert("P")
    width, height = image.size
    data = []

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            pixel = (pixel >> i) << i
            data.append(pixel)
    return data


def LZW(image_path):
    # 加载BMP图像
    image_path = image_path
    image = Image.open(image_path)

    # 将图像数据转换为八比特序列
    data = convert_image_to_data(image)
    data = "".join(str(num) for num in data)

    # 初始化字典
    dict_size = 128
    dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])

    compression_ratio = calculate_compression_ratio_2(image, result)

    print(image_path + " LZW Compression Ratio: {:.2f}".format(compression_ratio))


def LZW_8bits(image_path, i):
    # 加载BMP图像
    image_path = image_path
    image = Image.open(image_path)

    # 将图像数据转换为八比特序列
    data = convert_image_to_data_2(image, i)
    data = "".join(str(num) for num in data)

    # 初始化字典
    dict_size = 128
    dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])

    compression_ratio = calculate_compression_ratio_2(image, result)

    print(image_path + " LZW Compression Ratio: {:.2f}".format(compression_ratio))

if __name__ == '__main__':
    huffman("CT-1.bmp")
    huffman("lenna_8.bmp")
    huffman("MRI.bmp")
    huffman("US.bmp")
    LZW("CT-1.bmp")
    LZW("lenna_8.bmp")
    LZW("MRI.bmp")
    LZW("US.bmp")
    LZW_8bits("CT-1.bmp", 5)
    LZW_8bits("lenna_8.bmp", 5)
    LZW_8bits("MRI.bmp", 5)
    LZW_8bits("US.bmp", 5)