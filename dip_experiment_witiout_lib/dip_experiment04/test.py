import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

pixel = 65
pixel = (pixel >> 4) << 4

print(pixel)