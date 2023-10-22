import time
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from argparse import ArgumentParser as argparse
from face_recognition import mtcnn_detection, facenet_embedding

app = Flask(__name__)

parser = argparse()
parser.add_argument('-H', '--host', default="0.0.0.0",
                    help='Host server')
args = parser.parse_args()

# Manage API
@app.route('/api/v1/face_detector', methods=["POST"])
def face_detector():
    #data = request.get_json()
    #img_bytes = data["img"]
    img_bytes = request.files["img"]

    # Read input image
    #img = Image.open(BytesIO(base64.b64decode(img_bytes.stream)))
    img = Image.open(img_bytes.stream)

    # Measure inference time
    t0 = time.time()
    # Detect face on image
    b_box, distance = mtcnn_detection(img)
    t1 = time.time()
    #print("\nBounding box: ", b_box)
    #print("Distance: ", distance, "\n")

    # Form output dict
    opt_dict = {
        "message" : {
            "execution_time": t1-t0,
            "face_detected": distance is not None,
            "bounding_box": b_box,
            "distance" : distance
        } ,
        "success": True
    }

    return jsonify(opt_dict), 201

@app.route('/api/v1/identify', methods=["POST"])
def identify():
    data = request.get_json()
    img_bytes = data["img"]

    # Read input image
    img = Image.open(BytesIO(base64.b64decode(img_bytes)))

    # Measure inference time
    t0 = time.time()
    # Get embedding
    f_vect = facenet_embedding(img)
    t1 = time.time()

    # Form output dict
    opt_dict = {
        "message" : {
            "execution_time": t1-t0,
            "f_vect_shape": [int(x) for x in f_vect.shape],
            "ID": None
        } ,
        "success": True
    }

    return jsonify(opt_dict), 201

app.run(debug=True, host=args.host)