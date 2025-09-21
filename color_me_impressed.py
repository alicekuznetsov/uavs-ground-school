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

# Calculate the histogram of the hue channel
hue_channel = hsv_img[0, :, 0]
hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])

# Find the peak hues above 30% of the maximum histogram value
hist_peaks = []
threshold = hist.max() * 0.3
for hue, occurrences in enumerate(hist):
    if occurrences > threshold:
        # Don't add a hue if its close to another hue
        print(hue)
        if len(hist_peaks) == 0 or abs(hist_peaks[len(hist_peaks) - 1] - hue) >= 10:
            hist_peaks.append(hue)
            print('Appended', hue)
        else:
            print('Modifying', hist_peaks[len(hist_peaks) - 1], 'to', hue )
            hist_peaks[len(hist_peaks) - 1] = hue
print("Peaks found at", hist_peaks)

hue_masks = []
for hue in hist_peaks:
    # Max & Min functions allow for wrapping 
    lower_bound =  (max(0, hue - 10), 50, 50)
    upper_bound = (min(179, hue + 10), 255, 255)
    hue_masks.append(cv2.inRange(hsv_img, lower_bound, upper_bound))
    

fig, axs = plt.subplots(ncols=2, nrows=1 + len(hue_masks), figsize=(5.5, 3.5), layout="constrained")
# Hide axis markers :D
for ax in axs.flat:
    ax.axis('off')
axs[0, 1].axis('on')  

# -- Hue Histogram :D -- 
axs[0, 1].plot(hist)
axs[0, 1].set_title("Hue Distribution")
axs[0, 1].set_xlabel("Hue")
axs[0, 1].set_ylabel("Occurrences")
axs[0, 1].set_xlim([0, 180])

axs[0, 0].set_title("Provided Image")
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# -- Plot histograms and masks -- 
i = 0
while i < len(hue_masks):
    afwadha = cv2.bitwise_and(img, img, mask=hue_masks[i])
    axs[1 + i, 0].imshow(cv2.cvtColor(afwadha, cv2.COLOR_BGR2RGB))
    axs[1 + i, 1].imshow(cv2.cvtColor(hue_masks[i], cv2.COLOR_BGR2RGB))
    i += 1

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()






