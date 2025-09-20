import cv2

filename = 'deltarune_art.png'

def show_img(name, object):
    ''' 
        Combines a common sequence for showing images, destroying all windows after any key is pressed.
    '''
    cv2.imshow(name, object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(filename, cv2.IMREAD_COLOR_BGR) 

# Resize image to half the size 
small_size = [img.shape[1] // 2, img.shape[0] // 2]
img = cv2.resize(img, small_size)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

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
show_img("OG", img)

show_img("Red Mask", red_mask)
show_img("Red Only Image", red_img)

show_img("Blue Mask", blue_mask)
show_img("Blue Only Image", blue_img)

show_img("Purple Mask", purple_mask)
show_img("Purple Only Image", purple_img)




