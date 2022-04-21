import numpy as np
from PIL import Image


class canny_detection:

    def __init__(self, img_path, kernel_size=5, sigma=1.4, t_max=120, t_min=40):
        self.img = np.array(Image.open(img_path).convert("L"))
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.t_max = t_max
        self.t_min = t_min

    # using guassian filter to smooth the image
    # input:
    #      original image: img (ndarray)
    #      the size of the filter: kernel_size (int)
    #      the value of sigma: sigma (float)
    # output:
    #      new_img (ndarray)
    def smooth(self, img, kernel_size, sigma):
        H, W = img.shape[0], img.shape[1]
        pad = int(kernel_size / 2)
        kernel = np.zeros([kernel_size, kernel_size])
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i][j] = np.exp(-1 * ((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)) / (
                        2 * np.pi * sigma ** 2)
        kernel /= np.sum(kernel)
        new_img = np.zeros([H, W])
        for i in range(pad, H - pad):
            for j in range(pad, W - pad):
                new_img[i][j] = np.sum(img[i - pad:i + pad + 1, j - pad: j + pad + 1] * kernel)
        return new_img

    # using sobel filter to compute the gradient and gradient_direction of the image
    # input:
    #      the processed image after smoothing: img (ndarray)
    # output:
    #      the direction of gradient: direction (ndarray)
    #      the gradient of the image: new_img (ndarray)
    def sobel_filter(self, img):
        H, W = img.shape[0], img.shape[1]
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        pad = 1
        new_img = np.zeros([H, W])
        direction = np.zeros([H, W])
        for i in range(pad, H - pad):
            for j in range(pad, W - pad):
                dx = np.sum(kernel_x * img[i - pad:i + pad + 1, j - pad:j + pad + 1])
                dy = np.sum(kernel_y * img[i - pad:i + pad + 1, j - pad:j + pad + 1])
                direction[i][j] = np.arctan(dy / (dx + 0.0001))
                if direction[i][j] <= 0: direction[i][j] += 1.8
                if direction[i][j] >= 0.9: direction[i][j] -= 0.9
                new_img[i][j] = np.sqrt(dx ** 2 + dy ** 2)
        return direction, new_img

    # non_maximum suppression for thinning the edge
    # input:
    #      the graident of image: img (ndarray)
    #      the direction of gradient: direction (ndarray)
    # output:
    #      image after nms: new_img (ndarray)
    def nms(self, img, direction):
        H, W = img.shape[0], img.shape[1]
        new_img = np.copy(img)
        pad = 1
        for i in range(pad, H - pad):
            for j in range(pad, W - pad):
                if img[i][j] == 0: continue
                if direction[i][j] == 0:
                    w = [1, 0]
                elif direction[i][j] < 0.9:
                    w = [1, 1]
                elif direction[i][j] == 0.9:
                    w = [0, 1]
                else:
                    w = [0, 0]
                if img[i + w[0]][j + w[1]] >= img[i][j]:
                    new_img[i][j] = 0
                else:
                    new_img[i + w[0]][j + w[1]] = 0
                if img[i - w[0]][j - w[1]] >= img[i][j]:
                    new_img[i][j] = 0
                else:
                    new_img[i - w[0]][j - w[1]] = 0
        return new_img

    # using a high threshold to get the strong edge, then use dfs
    # to search the neighbors that are higher than the low threshold.
    # input:
    #      image after nms: im (ndarray)
    #      high threshold: t_max(high threshold)
    #      low threshold: t_min(low threshold)
    # output:
    #      final output: new_im (ndarray)
    def double_thresholding(self, im, t_max, t_min):
        H, W = im.shape[0], im.shape[1]
        strong = []
        visited = np.zeros_like(im)
        new_im = np.zeros_like(im)
        for i in range(H):
            for j in range(W):
                if im[i, j] >= t_max:
                    strong.append([i, j])

        def dfs(x, y):
            if x < 0 or x >= H or y < 0 or y >= W or visited[x][y] == 1:
                return
            if im[x, y] >= t_min:
                new_im[x, y] = 255
                visited[x, y] = 1
                dfs(x + 1, y + 1)
                dfs(x - 1, y - 1)
                dfs(x - 1, y + 1)
                dfs(x + 1, y - 1)
                dfs(x - 1, y)
                dfs(x + 1, y)
                dfs(x, y - 1)
                dfs(x, y + 1)
            else:
                visited[x, y] = 1
                new_im[x, y] = 0

        for i in strong:
            dfs(i[0], i[1])
        return new_im

    def detect(self):
        im = self.img
        smoothed_img = self.smooth(im, kernel_size=self.kernel_size, sigma=self.sigma)
        direction, gradient_img = self.sobel_filter(smoothed_img)
        nms_img = self.nms(gradient_img, direction)
        final_output = self.double_thresholding(nms_img, self.t_max, self.t_min)
        new_im = Image.fromarray(final_output)
        new_im.show()
