import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
# linear algebra library for SVD
import scipy.linalg as ln
import os
import math
import cv2



def svd_operation(path, filename):

    sample_image = img.imread(path)
    # Normalize the intensity values in each pixel by dividing it by 255
    sample_image = sample_image / 255

    # Now let check the space required/needed to store the image before we compress
    originalBytes = sample_image.nbytes
    # print("The space (in bytes) needed to store this image is", originalBytes, 'bytes')
    # print("The space (in kilo bytes) needed to store this image is", originalBytes / 1024, 'KB')
    # print("The space (in mega bytes) needed to store this image is", originalBytes / 1024 / 1024, 'MB')

    # Break the image into three different arrays based on colors
    red_image = sample_image[:, :, 0]
    green_image = sample_image[:, :, 1]
    blue_image = sample_image[:, :, 2]

    U_r, D_r, VT_r = np.linalg.svd(red_image, full_matrices = True)
    U_g, D_g, VT_g = np.linalg.svd(green_image, full_matrices = True)
    U_b, D_b, VT_b = np.linalg.svd(blue_image, full_matrices = True)

    bytesToBeStored = sum([matrix.nbytes for matrix in [U_r, D_r, VT_r, U_g, D_g, VT_g, U_b, D_b, VT_b]])
    # print("The matrices that we store have total size (in bytes):", bytesToBeStored, 'bytes')

    k = 1000

# Selecting k columns from U matrix and k rows from VT matrix
    U_r_k  = U_r[:, 0:k]
    VT_r_k = VT_r[0:k, :]
    U_g_k  = U_g[:, 0:k]
    VT_g_k = VT_g[0:k, :]
    U_b_k  = U_b[:, 0:k]
    VT_b_k = VT_b[0:k, :]

    D_r_k = D_r[0:k]
    D_g_k = D_g[0:k]
    D_b_k = D_b[0:k]

    compressedBytes = sum([matrix.nbytes for matrix in [U_r_k, D_r_k, VT_r_k, U_g_k, D_g_k, VT_g_k, U_b_k, D_b_k, VT_b_k]])

    # Reconstruct matrices for each color
    red_image_approx = np.dot(U_r_k, np.dot(np.diag(D_r_k), VT_r_k))
    green_image_approx = np.dot(U_g_k, np.dot(np.diag(D_g_k), VT_g_k))
    blue_image_approx = np.dot(U_b_k, np.dot(np.diag(D_b_k), VT_b_k))

    # Reconstruct the original image from color matrices
    imageReconstructed = np.zeros((sample_image.shape[0], sample_image.shape[1], 3))

    imageReconstructed[:, :, 0] = red_image_approx
    imageReconstructed[:, :, 1] = green_image_approx
    imageReconstructed[:, :, 2] = blue_image_approx

    # Correct the pixels where intensity value is outside the range [0,1]
    imageReconstructed[imageReconstructed < 0] = 0
    imageReconstructed[imageReconstructed > 1] = 1

    cv2.imwrite('./static/predict/{}'.format(filename))
    return imageReconstructed


    # print("The compressed matrices that we store now have total size (in bytes):", compressedBytes, 'bytes')
    # print("The compressed matrices that we store now have total size (in KB):", compressedBytes/1024, 'KB')
    # print("The compressed matrices that we store now have total size (in MB):", compressedBytes/1024/1024, 'MB')

    # ratio = compressedBytes / originalBytes * 100
    # print("\nThe compression ratio between the original image size and the compressed image is %.2f%s" %  (ratio, '%'))

