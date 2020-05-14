# -2-D-FFT-and-convolution
This repository contains the personalized implementation of my 2D FFT algorithm and 2D convolution implemented from scratch 

Problem 1 (50 points): Two-dimensional convolution.
=====================================================
(a) Write a function to implement 𝑔 = 𝑐𝑜𝑛𝑣2(𝑓,𝑤, 𝑝𝑎𝑑), where
𝑓 is an input image (grey, or RGB), 𝑤 is a 2-D kernel (e.g., 3 × 3 box
filter), and 𝑝𝑎𝑑 represents the 4 padding type (see page 17 in the handout
notes of lecture 8): clip/zero-padding, wrap around, copy edge, and
reflect across edge, as illustrated in the following example (2nd to 5th
row). The padding needs to be implemented by your code. You could use
built-in functions to compare results to check your implementation.


Test your function with the provided lenna.png and wolves.png 
For the kernel 𝑤, test the following:
1) Box filter
2) The simple first order derivative filter
3) And the filters : Prewitt, Sobel and Roberts filters.

(b) Create a grey image of size 1024x1024 pixels that consists of
a unit impulse at the center of the image (512, 512) and zeros elsewhere.
Use this image and a kernel of your choice (e.g., selected one from (a)) to
confirm that your function is indeed performing convolution. Show your
filter result and explain why your function is indeed performing
convolution.



Problem 2 (50 points): Implementing and testing the 2-D FFT and its inverse
using a built-in 1-D FFT algorithm.
==============================================================================
(a) [30 points] obtain a built-in routine that computes the 1-D FFT. For
example, if you know how to integrate c/c++ functions to Matlab or Python,
you can use open source implementations of 1-D FFT at www.fftw.org. Or
you can use fft in Matlab itself. Use the built-in 1-D FFT to implement 𝐹 =
𝐷𝐹𝑇2(𝑓) from scratch, where 𝑓 is an input grey image.

Test your function with the provided lenna.png and wolves.png (you can use
built-in color conversion functions to convert them to grey images). Before
apply the DFT2, you need to scale the grey image to the range [0, 1] (e.g.,
implement Problem 4 in HW02 and integrate it in your DFT2 function).
Visualize the spectrum and phase angle image. When visualizing them,
apply the transform 𝑠 = log (1 + 𝑎𝑏𝑠(𝐹)) or others as you see fit.



(b) Using your DFT2 to implement the inverse FFT of an input
transform 𝐹, 𝑔 = 𝐼𝐷𝐹𝑇2(𝐹) from scratch. (Hint, check item 12) in the top
of page 12 in the handouts of lecture 13).
Test your function for the two results in (a). Given an input grey image 𝑓,
you use (a) DFT2 to compute its 𝐹 , and use your IDFT2 to compute 𝑔 from
𝐹. Visualize 𝑓 and 𝑔 and they should look identical. To double-check it,
visualize 𝑑 = 𝑓 − 𝑔, which should be a black image.

