from flask import Flask, request, render_template, send_file, send_from_directory
import numpy as np
import cv2
from scipy.linalg import svd
import base64
import os
import matplotlib.image as img
from io import BytesIO
from werkzeug.utils import secure_filename
import requests
import urllib.request
import uuid
from urllib.parse import urlparse
import time
import random
import string


app = Flask(__name__)
selected_image = None
app.config['UPLOAD_FOLDER'] = './static/upload'
app.config['DOWNLOAD_FOLDER'] = './static/predict'

filename = None

@app.route('/')
def index():
    return render_template('compress.html')

@app.route('/compress', methods=['POST'])
def compress():
    # if request.method == 'POST':
    global selected_image
    global upload_path
    global filename
    global compressed_image_filename
    global compressed_image_path
    global compressed_img


     # Check if the user selected the "Upload Image" option
    if request.form.get('input_option') == 'upload':

        if 'image' in request.files:
            # Get the uploaded image file
            file = request.files['image']
            if file.filename == '':
                return 'No file selected', 400


            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Read the image using OpenCV
            # img = cv2.imread(upload_path)
            img = cv2.imread(upload_path)
        

            # Store the selected image
            selected_image = img

    else:  # User selected the "Image Link" option
        # Get the image link from the form
        image_link = request.form.get('image_link')

        filename = secure_filename(str(uuid.uuid4()) + '.jpg')  # Change the filename as needed
        
        # Save the image from the link to your PC
        urllib.request.urlretrieve(image_link, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Update the selected_image and upload_path variables
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        

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



     # Generate a unique filename for the compressed image
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    compressed_image_filename = f"compressed_{time.time()}_{random_string}.jpg"
    compressed_image_path = os.path.join(app.config['DOWNLOAD_FOLDER'], compressed_image_filename)


    cv2.imwrite(compressed_image_path, compressed_img)

    # cv2.imwrite(upload_path, compressed_img)

    # compress = compressed_img.nbytes
    now = round(selected_image.nbytes, 3)
    noow = round(now/ 1024, 3)
    new = round(now/1024/1024, 3)



    # Get the file size of the compressed image in megabytes
    compressed_file_size_b = round(len(compressed_img_encoded), 3)
    compressed_file_size_kb = round(len(compressed_img_encoded) / (1024), 3)
    compressed_file_size_mb = round(len(compressed_img_encoded) / (1024 * 1024), 3)

    # Difference
    byte_diff = round(now - compressed_file_size_b, 3)
    kb_diff = round(noow - compressed_file_size_kb, 3)
    mb_diff = round(new - compressed_file_size_mb, 3)

    # Return the compressed image and selected image to the template
    return render_template('compress.html', new = new, compressed_image=compressed_img_base64, 
                            selected_image=selected_img_base64, k=k, 
                            now = now, noow = noow,
                            compressed_file_size_b = compressed_file_size_b,
                            compressed_file_size_kb = compressed_file_size_kb,
                            compressed_file_size_mb = compressed_file_size_mb,
                            byte_diff = byte_diff, kb_diff = kb_diff, mb_diff = mb_diff
                        )
    # return render_template('index.html', upload = False)


@app.route('/download')
def download():
    global compressed_image_path
    
    # Replace this placeholder code with your own logic to get the path of the compressed image
    
    if os.path.exists(compressed_image_path):
        return send_file(compressed_image_path, as_attachment=True)
    else:
        return 'Compressed image not found', 404







if __name__ == '__main__':
    app.run(debug = True)
