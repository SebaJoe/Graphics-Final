from scipy import ndimage, misc
import imageio


f = misc.imread('scissors.jpg')

def local_cost(p, q):
    return 0

import matplotlib.pyplot as plt

plt.imshow(ndimage.gaussian_gradient_magnitude(f, sigma=1))
plt.show()