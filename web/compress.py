from flask import Flask, request, render_template
import numpy as np
import cv2
from scipy.linalg import svd
import base64
import os
import matplotlib.image as img


def compress():
    global selected_image

    # Check if an image file was uploaded
    if 'image' in request.files:
        # Get the uploaded image file
        file = request.files['image']
        
        # Read the image using OpenCV
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Store the selected image
        selected_image = img
    
    # Perform SVD on the image channels (R, G, B)
    img_r, img_g, img_b = cv2.split(selected_image)
    U_r, S_r, V_r = svd(img_r)
    U_g, S_g, V_g = svd(img_g)
    U_b, S_b, V_b = svd(img_b)

    # Get the value of k from the form
    k = int(request.form.get('k', 100))  # Default value of 100 if 'k' is not provided

    # Perform compression by keeping only a subset of singular values
    compressed_img_r = U_r[:, :k] @ np.diag(S_r[:k]) @ V_r[:k, :]
    compressed_img_g = U_g[:, :k] @ np.diag(S_g[:k]) @ V_g[:k, :]
    compressed_img_b = U_b[:, :k] @ np.diag(S_b[:k]) @ V_b[:k, :]

    # Merge the compressed channels back into a color image
    compressed_img = cv2.merge((compressed_img_r, compressed_img_g, compressed_img_b))

    

    # Encode the compressed image as base64
    _, img_encoded = cv2.imencode('.jpg', compressed_img)
    compressed_img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Encode the selected image as base64
    _, selected_img_encoded = cv2.imencode('.jpg', selected_image)
    selected_img_base64 = base64.b64encode(selected_img_encoded).decode('utf-8')

    # Encode the compressed image as base64
    _, compressed_img_encoded = cv2.imencode('.jpg', compressed_img)
    compressed_img_base64 = base64.b64encode(compressed_img_encoded).decode('utf-8')

    return compressed_img_base64, selected_img_base64

    # compress = compressed_img.nbytes
    # now = selected_image.nbytes
    # print('compressMB', compress/1024/1024)
    # print('originalMB', now/1024/1024)


    
    # Get the file size of the original image in megabytes
    # original_file_size_mb = os.path.getsize(file.filename) / (1024 * 1024)

    # Get the file size of the compressed image in megabytes
    compressed_file_size_mb = len(compressed_img_encoded) / (1024 * 1024)

    # print('Before: ', original_file_size_mb)
    # print('After: ', compressed_file_size_mb)
    # Return the compressed image and selected image to the template
    # return render_template('compress.html', compressed_image=compressed_img_base64, selected_image=selected_img_base64, k=k)



