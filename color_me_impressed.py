import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img(name, object):
    ''' 
        Combines a common sequence for showing images, destroying all windows after any key is pressed.
    '''
    cv2.imshow(name, object)
    cv2.moveWindow(name, 32, 55) # Make sure the window is in frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- Reading the file to analyze ----------

filename = input("Please provide the full name (name.filetype) of the image to analyze the assets folder: ")
filepath = 'assets/' + filename

img = cv2.imread(filepath) 

if img is None or not img.any():
    print("Improper filename provided, using default image")
    filename = 'wallpaper.jpeg'
    img = cv2.imread('assets/' + filename) 

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

# ---------- Masking out the peak hues ----------

# Calculate the histogram of the hue channel
hues, s, v = cv2.split(hsv_img)
hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])

hist_peaks = []
threshold = hist.mean() * 1.25
hue_range = 15

# Create hist_peaks with only hues above the threshold
hist_peaks = [hue for hue, occurrences in enumerate(hist) if occurrences > threshold]
print("Peaks found at", hist_peaks)

# Combine similar peaks
i = 0
while i < len(hist_peaks):
    last_peak = hist_peaks[i-1]
    cur_peak = hist_peaks[i]

    # 2 Peaks are the "same color"
    if abs(cur_peak - last_peak) < hue_range:
        new_peak = (cur_peak + last_peak) // 2
        hist_peaks[i] = new_peak
        hist_peaks.remove(last_peak)
    # Same deal, but considers the wrap around nature of the circle
    elif (cur_peak + 180) - last_peak < hue_range * 2:
        hist_peaks.remove(cur_peak)
    else:
        i += 1
print("Peaks Modified to", hist_peaks)

# Creating masks :D
hue_masks = []
for hue in hist_peaks:
    # Check for wrapping around the 180 (top) boundary
    if hue + hue_range > 180:
        # Create a mask for the top part of the range (e.g., 170-180)
        lower_bound1 = (hue - hue_range, 0, 0)
        mask1 = cv2.inRange(hsv_img, lower_bound1, (180, 255, 255))

        # Create a mask for the bottom part of the range (e.g., 0-10)
        upper_bound2 = (hue + hue_range - 180, 255, 255)
        mask2 = cv2.inRange(hsv_img, (0, 0, 0), upper_bound2)

        # Combine the two masks using a bitwise OR
        hue_masks.append(cv2.bitwise_or(mask1, mask2))

    # Check for wrapping around the 0 (bottom) boundary
    elif hue - hue_range < 0:
        # Create a mask for the bottom part of the range (e.g., 0-10)
        upper_bound1 = (hue + hue_range, 255, 255)
        mask1 = cv2.inRange(hsv_img, (0, 0, 0), upper_bound1)

        # Create a mask for the top part of the range (e.g., 170-180)
        lower_bound2 = (180 + hue - hue_range, 0, 0)
        mask2 = cv2.inRange(hsv_img, lower_bound2, (180, 255, 255))

        # Combine the two masks
        hue_masks.append(cv2.bitwise_or(mask1, mask2))

    # Normal case: no wrapping :p
    else:
        lower_bound = (hue - hue_range, 0, 0)
        upper_bound = (hue + hue_range, 255, 255)
        hue_masks.append(cv2.inRange(hsv_img, lower_bound, upper_bound))
    
# ---------- Actually showing the content! ----------

fig, axs = plt.subplots(ncols=2, nrows=1 + len(hue_masks), figsize=(5.5, 3.5), layout="constrained")
fig.canvas.manager.set_window_title('Hue Masks of ' + filename)

# Hide axis markers
for ax in axs.flat:
    ax.axis('off')
axs[0, 1].axis('on')  

# Hue Histogram
axs[0, 1].plot(hist)
axs[0, 1].set_title("Hue Distribution")
axs[0, 1].set_xlabel("Hue")
axs[0, 1].set_ylabel("Occurrences")
axs[0, 1].set_xlim([0, 180])

axs[0, 0].set_title("Provided Image")
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Place images and masks on our window
i = 0
while i < len(hue_masks):
    masked_img = cv2.bitwise_and(img, img, mask=hue_masks[i])
    axs[1 + i, 0].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    axs[1 + i, 1].imshow(cv2.cvtColor(hue_masks[i], cv2.COLOR_BGR2RGB))
    axs[1 + i, 1].set_title(hist_peaks[i])
    i += 1

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



