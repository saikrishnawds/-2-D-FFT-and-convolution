
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve(image, boxfilter):
    
    (iH, iW) = image.shape[:2]
    (kH, kW) = boxfilter.shape[:2]

    
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                    
                    roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                    
                    k = (roi * boxfilter).sum()

                    
                    output[y - pad, x - pad] = k

    
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

   
    return output


# using the box filter to see effects of convolution of a 1024x1024 image with impulse at the center.
boxfilter = np.array((
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9],
	[1/9, 1/9, 1/9]))






X = np.random.random((1024, 1024)) # sample B&W image of size 1024 x 1024
plt.imshow(X, cmap="gray")


print(X.shape)

for i in range(0,1024):
    for j in range(0,1024):
        if i==512 and j==512:
            X.itemset(i,j,255)
        else:
            X.itemset(i,j,0)
        

cv2.imshow("Black image with impulse in center",X)
print(X.shape)


   
convoleOutputX = convolve(X, boxfilter)

cv2.imshow("X img conv", convoleOutputX)









 



