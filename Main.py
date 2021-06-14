import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from Functions import *
import os
import json
from matplotlib.widgets import RectangleSelector

# Specify the stain to be investigated: Lyso, Mito or CellMask
stain = 'Mito'

# To store results set variable to True
store_results = 0
filename = 'venv/CoLoc_Zn_Mito2hours.json'

img1_string = 'Images/IncubationTimes/24hours/PS+Mito/24mars-24h-incu_PS+M5_ch01.tif'   # Image of PS
img2_string = 'Images/IncubationTimes/24hours/PS+Mito/24mars-24h-incu_PS+M5_ch00.tif'   # Image of stain

red_channel = plt.imread(img1_string)
green_channel = plt.imread(img2_string)
blank_channel = np.zeros((len(red_channel), len(red_channel)))

# split the channels into two images
image1 = red_channel
image2 = green_channel

# Normalize
image1 = image1 / 255
image2 = image2 / 255

# Calculate initial PCC
start_pcc = newPCC(image1, image2)
print(start_pcc)

# Choose thresholding method: Costes or Otsu
thresh_method = 'Costes'
thresh1, thresh2 = 0,0

if (start_pcc < 0.02) and (thresh_method == 'Costes'):      # if PCC<0.02, average intenisty of manual drawn roi
                                                            # is used as threshold

    def line_select_callback(eclick,erelease):
        global thresh1
        global thresh2
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        print(x1,x2)
        print(y1,y2)
        thresh1 = get_mean_of_nonzero(image1[y1:y2, x1:x2])
        thresh2 = get_mean_of_nonzero(image2[y1:y2, x1:x2])
        print(thresh1,thresh2)

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key == 't':
            if toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            else:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)


    fig, ax = plt.subplots()
    ax.imshow(image1)

    rs = RectangleSelector(ax, line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # disable middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()

elif thresh_method == 'Costes':
    # Calculate threshold by Costes method
    thresh1,thresh2 = Costes(image1,image2,0.2)

elif thresh_method == 'Otsu':
    # Calculate threshold from Otsu method
    plt.figure('hist1')
    plt.hist(flatten(image1), bins=256)
    plt.figure('hist2')
    plt.hist(flatten(image2), bins=256)
    thresh1 = Otsu(image1)
    thresh2 = Otsu(image2)

print(thresh1,thresh2)

# Subtract median from region around pixel. Number indicates nxn region
image2_med = image2
image1_med = image1
if stain == 'Mito' or stain == 'Lyso' or stain == 'CellMask':
    image2_med = median_filter(image2,25)
    image1_med = median_filter(image1,25)

# Apply threshold
image1_thresh = (threshold(image1_med,thresh1) > 0) * image1
image2_thresh = (threshold(image2_med,thresh2) > 0) * image2

fig,ax = plt.subplots(1, figsize=(4, 4))

# Set whitespace to 0
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Create binary mask
if stain == 'CellMask':

    bin_mask = CellMask_roi(img2_string,1,30)
    roi_size = len(np.nonzero(bin_mask)[0])    # number of pixels in ROI
    fig_roi = plt.figure('binary mask', figsize=(4,4))
    fig_roi.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(bin_mask, cmap='gray')
    print(roi_size)
else:
    bin_mask = roi_morphological(image2)
    fig_roi = plt.figure('binary mask', figsize=(4,4))
    fig_roi.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(bin_mask, cmap='gray')
    roi_size = len(np.nonzero(bin_mask)[0])    # number of pixels in ROI
    print(roi_size)

# Multiply images with mask
image1_thresh = image1_thresh * bin_mask
image2_thresh = image2_thresh * bin_mask

# Find nonzero values
image1_thresh_nonzero = np.nonzero(image1_thresh)
image2_thresh_nonzero = np.nonzero(image2_thresh)

image1_nonzero_value = len(image1_thresh_nonzero[0])
image2_nonzero_value = len(image2_thresh_nonzero[0])

# Calculate excpected MCC values for later use in statistics
M1_expected = image2_nonzero_value / roi_size
M2_expected = image1_nonzero_value / roi_size



# multiplied thresholded images to find overlap
multiplied = np.multiply(image1_thresh, image2_thresh)

multiplied_nonzero = np.nonzero(multiplied)
multiplied_value = len(multiplied_nonzero[0])

# Overlap of pixels with intensities higher than 0 after threshold
overlap = multiplied_value / (image1_nonzero_value + image2_nonzero_value - multiplied_value)


# Calculate MCC coefficients M1 and M2
M1,M2 = MCC(image1_thresh, image2_thresh)

# Show overlapping region as binary image
image1_binary = image1_thresh > 0
image2_binary = image2_thresh > 0
image_overlap = image1_binary * image2_binary

fig,ax = plt.subplots(1, figsize=(4, 4))

# Set whitespace to 0
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Display the image
ax.imshow(image_overlap, cmap='gray')

# Turn off axes and set axes limits
ax.axis('tight')
ax.axis('off')

#Calculate SNR
signal_mean = (get_mean_of_nonzero(image1_thresh) + get_mean_of_nonzero(image2_thresh))/2
noise_mean = (get_mean_of_nonzero(inverse_threshold(image1, thresh1)) + get_mean_of_nonzero(inverse_threshold(image2,thresh2)))/2
SNR = signal_mean/noise_mean


#Results
pcc = newPCC(image1_thresh,image2_thresh)
results = [pcc,M1,M2,overlap,thresh1,thresh2, M1_expected, M2_expected, img1_string]
results_string = ['PCC', 'M1', 'M2', 'Overlap', 'T1','T2', 'M1 expected', 'M2 expected', 'File']
print('final T1 = ', thresh1, '\nfinal T2 = ', thresh2)
print('M1 = ', M1)
print('M2 = ', M2)
print('Overlap = ', overlap)
print('final PCC = ', pcc)
print('SNR = ', SNR)
print('M1 expected = ', M1_expected)
print(('M2 expected = ', M2_expected))

if store_results:
    #Save results in file as a dictionary
    file = open(filename)
    results_file = json.load(file)
    j = 0
    for i in results_string:
        results_file[i].append(results[j])
        j += 1
    file.close()
    file = open(filename,'w')
    json.dump(results_file,file)
    file.close()


#Show images before and after threshold

fig1 = plt.figure('red image before threshhold', figsize=(4,4))
fig1.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.imshow(image1)

fig2 = plt.figure('blue image before threshold', figsize=(4,4))
fig2.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.imshow(image2)

fig3 = plt.figure('red image after threshold', figsize=(4,4))
fig3.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.imshow(image1_thresh)

fig4 = plt.figure('blue image after threshold', figsize=(4,4))
fig4.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.imshow(image2_thresh)

fig5 = plt.figure('Image after threshold', figsize=(4,4))
fig5.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.imshow(np.dstack((image1_thresh,image2_thresh,blank_channel)))

plt.show()


