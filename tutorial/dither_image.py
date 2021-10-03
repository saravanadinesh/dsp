""" dither_image.py - Demonstrates usefulness of dithering while
    quantizing gray scale image to black and white. """

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt


img = io.imread('Albert_Einstein_Head.jpg')
meanval = np.mean(img)

# Convert to B&W
bw_img = np.empty(shape=img.shape)
bw_img[img<meanval] = 0     # Set pixels below mean pixel value to black,
bw_img[img>=meanval] = 255  # and the rest to white

# Add dither and convert to B&W
dither = np.random.uniform(low=-meanval, high =meanval, size = img.shape)
dithered_img = img + dither

bw_img_post_dither = np.empty(shape=img.shape)
bw_img_post_dither[dithered_img<meanval] = 0     # Set pixels below mean pixel value to black,
bw_img_post_dither[dithered_img>=meanval] = 255  # and the rest to white

# Upsample by 4x, add dither and convert to B&W
img_4x = np.kron(img, np.ones((4,4))) 
dither_4x = np.random.uniform(low=-meanval, high =meanval, size = img_4x.shape)
dithered_img_4x = img_4x + dither_4x

bw_img_4x_post_dither = np.empty(shape=img_4x.shape)
bw_img_4x_post_dither[dithered_img_4x<meanval] = 0     # Set pixels below mean pixel value to black,
bw_img_4x_post_dither[dithered_img_4x>=meanval] = 255  # and the rest to white


# Visualisation
fig1, ax1 = plt.subplots()
ax1.imshow(img, cmap='gray')
fig1.suptitle('Original')

fig2, ax2 = plt.subplots()
ax2.imshow(bw_img, cmap='gray')
fig2.suptitle('Black and white (without using dither)')

fig3, ax3 = plt.subplots()
ax3.imshow(bw_img_post_dither, cmap='gray')
fig3.suptitle('Black and white (using dither)')

fig4, ax4 = plt.subplots()
ax4.imshow(img_4x, cmap='gray')
fig4.suptitle('Orignal image with 4x OSR')

fig5, ax5 = plt.subplots()
ax5.imshow(bw_img_4x_post_dither, cmap='gray')
fig5.suptitle('Black and white (using 4x OSR and dither)')

plt.show()
