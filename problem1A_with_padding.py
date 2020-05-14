

from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve(image, kernel,pad):
    # taking the spatial dimensions of the image, along with the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Implementing the padding using the user input

        if pad==1:
            image=zeropad(image)
        elif pad==2:
            image=borderreplicate(image)
        elif pad==3:
            image=borderreplicate(image)  # wrap around edge and border replicate are the same for 1 layer of padding
        else:
            image=reflectacross(image)

        


    #Creating a zero pixel image , and then updating the  output of each convolution into this zero pixel image
    output = np.zeros((iH, iW), dtype="float32")

    # looping over the input image, "sliding" the kernel across each (x, y)-coordinate from left-to-right and top to bottom
   
    for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                    # extracting the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
                     
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                    # perform the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
                    
                    k = (roi * kernel).sum()

                    # storing the convolved value in the output (x,y)- coordinate of the output image
                     
                    output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
        # Deleting the padded boundary after convolution:

    # we can just make the boundary pixels blank or white, as shown in the project problem.

    imgarp=output # where imgarp is the final output image after removing padding

    for k in range(0,3):
        for j in range(1,c+3):
            imgarp.itemset((1,j,k),255)
            imgarp.itemset((r+2,j,k),255)     # deleting the padded rows

    for k in range(0,3):
        for i in range(1,r+3):
            imgarp.itemset((i,1,k),255)   
            imgarp.itemset((i,1,k),255)        # deleting the padded columns

    cv2.imshow("Final image after padding removed",imgarp)
        
    # return the output image
    return imgarp



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

print(img.shape)
r=539
c=1500
"""
img=cv2.imread('lena.png')    # Uncomment this region and change the values of the r and c if the input image is different
print(img.shape)
r=440
c=440
"""


#Zero padding
def zeropad(img):
    X1 = np.random.random((r+2, c+2,3))          # creating a black image of size (r+2) x (c+2), where r, c are the number of rows and columns in the given image.
    for k in range(0,3):
        for i in range(0,r+2):
            for j in range(0,c+2):
                X1.itemset((i,j,k),0)


    for k in range(0,3):
        for i in range(2,r+1):
            for j in range(2,c+1):
                X1[i,j,k]=img[i-1,j-1,k]
    cv2.imshow('zero-padded image',img)          # X1 is the zero padded image, and we take that as the model first and proceed with the other types of padding.
    return img

#Border replicate padding and wrap around padding
def borderreplicate(img):
    X1 = np.random.random((r+2, c+2,3))          # creating a black image of size (r+2) x (c+2), where r, c are the number of rows and columns in the given image.
    for k in range(0,3):
        for i in range(0,r+2):
            for j in range(0,c+2):
                X1.itemset((i,j,k),0)


    for k in range(0,3):
        for i in range(2,r+1):
            for j in range(2,c+1):
                X1[i,j,k]=img[i-1,j-1,k]
    cv2.imshow('zero-padded image',img)          # X1 is the zero padded image, and we take that as the model first and proceed with the other types of padding.
    X2=X1
    for k in range(0,3):
        for j in range(2,c+2):
            X2[1,j,k]=img[1,j-1,k]
            X2[r+2,j,k]=img[r,j-1,k]   # padding the lower and upper row with copy edge method
    for k in range(0,3):                 # this is the same as wrap around padding when we consider only one layer of padding of rows and columns 
        for i in range(1,r+3):
            X2[i,c+2,k]=X2[i,c+1,k]
            X2[i,1,k]=X2[i,2,k]          # X2 is the copy across padded image, and we can reflect the first and last columns of this image and  
    cv2.imshow("Copy edge padding",X2)   # we can obtain the reflect across the edge padding.

    return X2

#Reflect across padding
def reflectacross(img):
    X1 = np.random.random((r+2, c+2,3))          # creating a black image of size (r+2) x (c+2), where r, c are the number of rows and columns in the given image.
    for k in range(0,3):
        for i in range(0,r+2):
            for j in range(0,c+2):
                X1.itemset((i,j,k),0)


    for k in range(0,3):
        for i in range(2,r+1):
            for j in range(2,c+1):
                X1[i,j,k]=img[i-1,j-1,k]
    cv2.imshow('zero-padded image',img)          # X1 is the zero padded image, and we take that as the model first and proceed with the other types of padding.
    X2=X1
    for k in range(0,3):
        for j in range(2,c+2):
            X2[1,j,k]=img[1,j-1,k]
            X2[r+2,j,k]=img[r,j-1,k]   # padding the lower and upper row with copy edge method
    for k in range(0,3):                 # this is the same as wrap around padding when we consider only one layer of padding of rows and columns 
        for i in range(1,r+3):
            X2[i,c+2,k]=X2[i,c+1,k]
            X2[i,1,k]=X2[i,2,k]          # X2 is the copy across padded image, and we can reflect the first and last columns of this image and  
    cv2.imshow("Copy edge padding",X2)   # we can obtain the reflect across the edge padding.
    X3=X2
    randrow=np.random.random((1,c+2,3))  # create a random one row image to store the values of the pixels of the first row and then exchange the values with the last row

    for k in range(0,3):
        for j in range(1,c+3):
            randrow[1,j,k]=X3[1,j,k]  
            X3[1,j,k]=X3[r+2,j,k]
            X3[r+2,j,k]=randrow[1,j,k]  



    randcol=np.random.random((r+2,1,3))  # create a random one column image to store the values of the pixels of the first column  and then exchange the values with the last row

    for k in range(0,3):                              
        for i in range(1,r+3):
            randcol[i,1,k]=X3[i,1,k] 
            X3[i,1,k]=X3[i,c+2,k]
            X3[i,c+2,k]=randcol[i,1,k]
    cv2.imshow("reflect across padding",X3)
    return X3

###############################################################################


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               # here
b_img=image[:,:,0]
g_img=image[:,:,1]
r_img=image[:,:,2]

# loop over the kernels
for (kernelName, kernel) in kernelBank:
        # applying the kernel to theimage using both the  `convolve` function and OpenCV's `filter2D` function
    
    p=int(input("Enter 1 for zero padding, 2 for border replicate padding, 3 for wrap around padding, and 4 for reflect across padding"))
    print(" applying {} kernel".format(kernelName))
    convoleOutputg = convolve(gray, kernel,p)
    convoleOutput1 = convolve(b_img, kernel,p)
    convoleOutput2 = convolve(g_img, kernel,p)
    convoleOutput3 = convolve(r_img, kernel,p)
##    opencvOutputg = cv2.filter2D(gray, -1, kernel)
##    opencvOutput1 = cv2.filter2D(b_img, -1, kernel)
##    opencvOutput2 = cv2.filter2D(g_img, -1, kernel)
##    opencvOutput3 = cv2.filter2D(r_img, -1, kernel)
##    
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
    
    rgbimg2=cv2.merge((convoleOutput1,convoleOutput2,convoleOutput3))
    cv2.imshow("{}-Final merge".format(kernelName),rgbimg2)



