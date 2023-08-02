# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

# 读取目标路径
image_factory = mpimg.imread('./factory.bmp')
image_PET_image = mpimg.imread('./PET_image.bmp')


def fft2d(img):
    """Compute the 2D FFT of a grayscale image"""
    rows, cols = img.shape
    # Compute the FFT of each row
    temp = np.empty((512, 512), dtype=np.complex128)
    for i in range(rows):
        temp[i, :] = fft(img[i, :])
    # Compute the FFT of each column
    result = np.empty((512, 512), dtype=np.complex128)
    for j in range(cols):
        result[:, j] = fft(temp[:, j])
    result = abs(result)
    print(result)
    for i in range(512):
        for j in range(512):
            result[i][j] = int(result[i][j])
            if result[i][j] > 255:
                result[i][j] = 255
            elif result[i][j] < 0:
                result[i][j] = 0
    print(result)
    plt.subplot(421)
    plt.imshow(result, cmap='gray')
    plt.title("fft")


def fft(x):
    """Compute the 1D FFT of x using Cooley-Tukey algorithm"""
    x = zero_fill(x)
    n = x.shape[0]
    if n == 1:
        return x
    else:
        even = fft(x[0::2])
        odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate([even + factor[:n//2] * odd, even + factor[n//2:] * odd])


def zero_fill(x):
    n = x.size
    m = n
    cnt = 0
    res = 1
    while res < n:
        res = res << 1
    if res == n:
        return x
    else:
        while m != 0:
            m = m // 2
            cnt += 1
        num = 2 ** cnt - n
        x = np.pad(x, (0, num), 'constant', constant_values=(0, 0))
        return x


def ifft2d(img_fft):
    """Compute the 2D IFFT of a grayscale image"""
    rows, cols = img_fft.shape
    # Compute the IFFT of each row
    temp = np.empty_like(img_fft, dtype=np.complex128)
    for i in range(rows):
        temp[i] = ifft(img_fft[i])
    # Compute the IFFT of each column
    result = np.empty_like(img_fft, dtype=np.complex128)
    for j in range(cols):
        result[:,j] = ifft(temp[:,j])
    # Scale the result and take the real part
    return np.real(result / (rows * cols))


def ifft(x):
    """Compute the 1D IFFT of x using Cooley-Tukey algorithm"""
    n = x.shape[0]
    if n == 1:
        return x
    else:
        even = ifft(x[::2])
        odd = ifft(x[1::2])
        factor = np.exp(2j * np.pi * np.arange(n) / n)
        return np.concatenate([even + factor[:n//2] * odd, even + factor[n//2:] * odd])


def fifft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = np.log(np.abs(fshift))

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    plt.subplot(421), plt.imshow(res, 'gray'), plt.title('Fourier Image')
    plt.subplot(422), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')


def ILPF(image, d0):
    img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if int(((i - rows / 2) ** 2 + (j - cols / 2) ** 2) ** 1/2) <= d0:
                fshift[i][j] = fshift[i][j]
            else:
                fshift[i][j] = 0
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    plt.subplot(423), plt.imshow(iimg, 'gray'), plt.title('ILPF')


def IHPF(image, d0):
    img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            if int(((i - rows / 2) ** 2 + (j - cols / 2) ** 2) ** 1/2) > d0:
                fshift[i][j] = fshift[i][j]
            else:
                fshift[i][j] = 0

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    plt.subplot(425), plt.imshow(iimg, 'gray'), plt.title('IHPF')


def BWLPF(image, d0, n):
    img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            d = ((i - rows / 2) ** 2 + (j - cols / 2) ** 2) ** 1 / 2
            fshift[i][j] *=  1 / (1 + (d / d0) ** (2 * n))

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    plt.subplot(424), plt.imshow(iimg, 'gray'), plt.title('BWLPF')


def BWHPF(image, d0, n):
    img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            d = ((i - rows / 2) ** 2 + (j - cols / 2) ** 2) ** 1 / 2
            if d != 0:
                fshift[i][j] *=  1 / (1 + (d0 / d) ** (2 * n))
            else:
                fshift[i][j] = 0

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    plt.subplot(426), plt.imshow(iimg, 'gray'), plt.title('BWHPF')


def HF_1(img, cutoff_freq, order):
    img_log = np.log1p(np.array(img, dtype="float") / 255)
    img_fft = np.fft.fft2(img_log)
    H = np.zeros_like(img_fft)
    rows, cols = img_fft.shape

    for i in range(rows):
        for j in range(cols):
            H[i, j] = (1 - np.exp(-order * ((i - rows / 2)**2 + (j - cols / 2)**2) / (2 * cutoff_freq**2)))

    img_filtered = np.real(np.fft.ifft2(img_fft * H))
    img_exp = np.expm1(img_filtered) * 255
    img_out = np.uint8(np.clip(img_exp, 0, 255))

    plt.subplot(427), plt.imshow(img_out, 'gray'), plt.title('HF_1')


def HF_2(img, cutoff_freq, order):
    img_log = np.log1p(np.array(img, dtype="float") / 255)
    img_fft = np.fft.fft2(img_log)
    H = np.zeros_like(img_fft)
    rows, cols = img_fft.shape

    for i in range(rows):
        for j in range(cols):
            H[i, j] = (1 - np.exp(-order * ((i - rows / 2)**2 + (j - cols / 2)**2) / (2 * cutoff_freq**2)))

    img_filtered = np.real(np.fft.ifft2(img_fft * H))
    img_exp = np.expm1(img_filtered) * 255
    img_out = np.uint8(np.clip(img_exp, 0, 255))

    plt.subplot(428), plt.imshow(img_out, 'gray'), plt.title('HF_2')


if __name__ == '__main__':
    fifft(image_factory)
    ILPF(image_factory, 250)
    BWLPF(image_factory, 250, 1)
    IHPF(image_PET_image, 250)
    BWHPF(image_PET_image, 250, 1)
    HF_1(image_factory, 20, 2)
    HF_2(image_PET_image, 20, 2)
    plt.tight_layout()
    plt.show()