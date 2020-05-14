

from skimage.exposure import rescale_intensity
import numpy as np
import cv2

def convolve(image, kernel):
    # taking the spatial dimensions of the image, along with the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocating memory for the output image, taking care to "pad" the borders of the input image so the spatial size is not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    #Creating a zero pixel image , and then updating the  output of each convolution into this zero pixel image
    output = np.zeros((iH, iW), dtype="float32")

    # looping over the input image, "sliding" the kernel across each (x, y)-coordinate from left-to-right and top to bottom
    for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                    # extracting the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                    # performing the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
                    k = (roi * kernel).sum()

                    # storing the convolved value in the output (x,y)- coordinate of the output image
                    output[y - pad, x - pad] = k

    # rescaling the output image intensity to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output



# defining the different kernels to be used 

boxfilter = np.array((
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9]))
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]))
sobelY = np.array((
	[1, 2, 1],
	[0, 0, 0],
	[-1, -2, -1]))
prewittX = np.array((
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]))
prewittY = np.array((
	[1, 1, 1],
	[0, 0, 0],
	[-1, -1, -1]))
rowder = np.array((
	[0, 0 ,0],
        [-1 ,1 ,0],
        [0 ,0 ,0]))
colder = np.array((
	[-1 ,0 ,0],
	[1 ,0 ,0],
        [0 ,0 ,0]))
robertX = np.array((
	[0, 1 ,0],
        [-1 ,0 ,0],
	[0, 0 ,0]))
robertY = np.array((
	[1, 0 ,0],
	[0, -1 ,0],
        [0 ,0 ,0]))



# constructing the kernel bank, a list of kernels that is being applied to the 'convolve' function and the inbuilt opencv function
kernelBank = (
    ("boxfilter",boxfilter),
    ("sobelX", sobelX),
    ("sobelY", sobelY),
    ("prewittX", prewittX),
    ("prewittY", prewittY),
    ("rowder",rowder),
    ("colder", colder),
    ("robertX", robertX),
    ("robertY", robertY)

)


image = cv2.imread("wolves.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               # here
b_img=image[:,:,0]
g_img=image[:,:,1]
r_img=image[:,:,2]

# looping over the kernels
for (kernelName, kernel) in kernelBank:
     # applying the kernel to theimage using both the  `convolve` function and OpenCV's `filter2D` function
    print("[INFO] applying {} kernel".format(kernelName))
    convoleOutputg = convolve(gray, kernel)
    convoleOutput1 = convolve(b_img, kernel)
    convoleOutput2 = convolve(g_img, kernel)
    convoleOutput3 = convolve(r_img, kernel)
    opencvOutputg = cv2.filter2D(gray, -1, kernel)
    opencvOutput1 = cv2.filter2D(b_img, -1, kernel)
    opencvOutput2 = cv2.filter2D(g_img, -1, kernel)
    opencvOutput3 = cv2.filter2D(r_img, -1, kernel)
    
    """
    convoleOutput = convolve(gray, kernel)                  # here
    opencvOutput = cv2.filter2D(gray, -1, kernel)           # here
    """

    # show the output images
    #cv2.imshow("original", image)                             #here
    cv2.imshow("{}-Gray img conv".format(kernelName), convoleOutputg)
    cv2.imshow("{}-Blue img conv".format(kernelName), convoleOutput1)
    cv2.imshow("{}-Green img conv".format(kernelName), convoleOutput2)
    cv2.imshow("{}-Red img conv".format(kernelName), convoleOutput3)
    #cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    rgbimg2=cv2.merge((convoleOutput1,convoleOutput2,convoleOutput3)) # Merging the R,G and B channels and displaying the combined RGB image
    cv2.imshow("{}-Final merge".format(kernelName),rgbimg2)
