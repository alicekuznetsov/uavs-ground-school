import cv2
import matplotlib.pyplot as plt
import numpy as np

#  TODO: allow for user to provide image 
filename = 'assets/deltarune_art.png'

def show_img(name, object):
    ''' 
        Combines a common sequence for showing images, destroying all windows after any key is pressed.
    '''
    cv2.imshow(name, object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(filename, cv2.IMREAD_COLOR_BGR) 
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

# TODO: Use color distribution (the video has a histogram, thats a pretty good idea) to automatically determine which colors to mask

# Define the range for red in HSV - hue range is [0,179] (360 / 2)
# Red appears twice in HSV, so we need 2 masks :p
lower_red1 = (0, 50, 50) 
upper_red1 = (10, 255, 255)
lower_red2 = (170, 50, 50)  
upper_red2 = (180, 255, 255)

lower_blue = (85, 50, 50) 
upper_blue = (125, 255, 255)

lower_purple = (125, 50, 50) 
upper_purple = (140, 255, 255)

# MASKS: Black-and-white images containing the colors.
# Create masks for both red ranges and combine them.
red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
purple_mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

# Use masks on original image
red_img = cv2.bitwise_and(img, img, mask=red_mask)
blue_img = cv2.bitwise_and(img, img, mask=blue_mask)
purple_img = cv2.bitwise_and(img, img, mask=purple_mask)


# Show the world what you've made :D
# TODO: Put these in a figure

fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(5.5, 3.5), layout="constrained")
# Hide axis markers :D
for ax in axs.flat:
    ax.axis('off')
axs[0, 1].axis('on')  

# Calculate the histogram of the hue channel
hue_channel = hsv_img[0, :, :]
hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])

# Plot the histogram
axs[0, 1].plot(hist, color='orange')
axs[0, 1].set_title("Hue Distribution")
axs[0, 1].set_xlim([0, 180])

axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 0].imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
axs[1, 1].imshow(cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB))
axs[2, 0].imshow(cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB))
axs[2, 1].imshow(cv2.cvtColor(blue_mask, cv2.COLOR_BGR2RGB))
axs[3, 0].imshow(cv2.cvtColor(purple_img, cv2.COLOR_BGR2RGB))
axs[3, 1].imshow(cv2.cvtColor(purple_mask, cv2.COLOR_BGR2RGB))

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()






