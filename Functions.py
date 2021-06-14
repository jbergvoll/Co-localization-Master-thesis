import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import scipy.ndimage as nd
import scipy.stats as stats
from scipy import linalg
import scipy.optimize
import cv2


def get_mean_of_nonzero(img): #Calculates mean of non-zero pixel values
    mask = np.ma.masked_equal(img,0)
    return np.mean(mask)

def get_len_of_nonzero(img1,img2):  #Calculates number of pixels with either green or red signal
    non_zero_img1 = len(np.nonzero(img1)[0])
    non_zero_img2 = len(np.nonzero(img2)[0])
    overlap = np.multiply(img1,img2)
    non_zero_overlap = len(np.nonzero(overlap)[0])

    return non_zero_img1 + non_zero_img2 - non_zero_overlap


def CellMask_roi(img, dilate_iterations, T):  # Create mask based on edgde detection from CellMask
    # Let's load a simple image with 3 black squares
    image = cv2.imread(img)
    cv2.waitKey(0)

    # Grayscale
    kernel1 = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)

    morph1 = cv2.erode(image, kernel1, iterations=1)
    morph2 = cv2.dilate(morph1, kernel2, iterations=dilate_iterations)

    gauss = cv2.GaussianBlur(morph2, (25, 25), 5)
    # gauss = cv2.dilate(gauss,kernel2, iterations=1)

    gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

    val, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    gray = thresh
    # Find Canny edges
    edged = cv2.Canny(gray, 10, T)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

    # green channel is mask
    (b, g, r) = cv2.split(image)
    g = threshold(g, 254)
    g = g > 0
    plt.figure('CellMaskROI', figsize=(4,4))
    plt.imshow(g, cmap='gray')
    plt.axis('off')
    plt.axis('tight')
    plt.show()

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return g


def roi_morphological(img): # Returns ROI of image
    # opening to remove speckles. Typically noise
    img_open = nd.binary_opening(img, np.ones((2, 2)))

    # closing to fill holes.
    img_close = nd.binary_closing(img_open, np.ones((9,9)))

    return img_close

def pearson(img1,img2):
    x = flatten(img1)
    y = flatten(img2)
    return stats.pearsonr(x,y)[0]

def Otsu(image):    #Returns threshold value based on Otsu method
    return filters.threshold_otsu(image)

def PCC(img1,img2):
    #calculates correlation coefficient between two images of equal size
    enum = 0
    denum1 = 0
    denum2 = 0

    avg1 = get_mean_of_nonzero(img1)
    avg2 = get_mean_of_nonzero(img2)

    for i in range(len(img1)):
        for j in range(len(img1[i])):
            if img1[i,j] != 0 and img2[i,j] != 0:
                enum += (img1[i][j] - avg1) * (img2[i][j] - avg2)
                denum1 += (img1[i][j] - avg1)**2
                denum2 += (img2[i][j] - avg2)**2

    correlation = (enum)/(np.sqrt((denum1)*(denum2)))
    return correlation

def threshold(img, thresh): #Returns image above threshold.
    img_bin = img > thresh
    return img *img_bin

def inverse_threshold(img,thresh):
    #returns image below threshold. This is used in Costes method.
    img_bin = img < thresh
    return img * img_bin

def flatten(img):     #flattens image into 1D array
    return np.ndarray.flatten(img)

def newPCC(img1,img2):
    #First change the images so only non-zero in both channels present
    x_bin = img1 > 0
    y_bin = img2 > 0
    overlap = x_bin*y_bin

    x = np.ma.masked_equal(flatten(img1*overlap),0)
    y = np.ma.masked_equal(flatten(img2*overlap),0)

    #Calculate PCC according to formula
    xmean = np.mean(x)
    ymean = np.mean(y)

    xm = x - xmean
    ym = y - ymean

    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)

    r = np.dot(xm / normxm, ym / normym)

    return r

def Costes(img1,img2,start_T):
    #incrementally decrease threshold until values below threshold have PCC = 0 (uncorrelated)

    #Find linear relationship between channels
    a,b = leastsquare(img1,img2)

    def func(x, a, b):  # Linear func to relate the two intensities
        return a + b * x

    pcc = newPCC(img1,img2)
    T1 = start_T
    T2 = func(start_T,a,b)
    print('T1 = ',T1,'\nT2 = ',T2)
    dI = 1/255   #decrement threshold
    print(pcc)
    while pcc > 0.02:   #This value may be changed for better result/stability.
        if (T1 - dI) < 0.01 or (T2 - dI) < 0.01:
            break
        if pcc == 1:
            break

        img1_thresh = inverse_threshold(img1,T1)
        img2_thresh = inverse_threshold(img2,T2)

        pcc = newPCC(img1_thresh,img2_thresh)

        T1 = T1 - dI
        T2 = func(T1,a,b)
        print('T1 = ', T1, '\nT2 = ', T2)
        print('PCC = ',pcc)

    return T1,T2


def leastsquare(img1,img2): # Calculates Least square fit of two channels, a + bx.
    def func(x,a,b):
        return a + b*x

    lstsq = scipy.optimize.curve_fit(func, flatten(img1), flatten(img2))
    a,b = lstsq[0][0], lstsq[0][1]
    return a,b

def median_filter(img,n): #Median subtraction filter
    return img - nd.median_filter(img,n)

def MCC(image1,image2): #Calculates M1 and M2.
    M1den = 0
    M2den = 0
    M1num = 0
    M2num = 0


    for i in range(len(image2)):
        for j in range(len(image2[0])):
            M1den += image1[i, j]
            M2den += image2[i, j]

            if image1[i, j] > 0 and image2[i, j] > 0:
                M1num += image1[i, j]
                M2num += image2[i, j]

    # Manders correlation coefficients.
    M1 = M1num / M1den
    M2 = M2num / M2den

    return M1,M2

def get_max_of_nonzero(img):
    mask = np.ma.masked_equal(img,0)
    return np.max(mask)