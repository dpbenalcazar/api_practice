# import the opencv library 
import cv2 
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from argparse import ArgumentParser as argparse

parser = argparse()
parser.add_argument('-ip', '--IP', default="192.168.11.23",
                    help='Server IP address')
parser.add_argument('-e', '--enroll', action="store_true",
                    help='Enroll new person')
parser.add_argument('-n', '--names', default="",
                    help='Name for enrolling new ID.')
parser.add_argument('-l', '--last_names', default="",
                    help='Last Name(S) for enrolling new ID.')
parser.add_argument('-b', '--date_of_birth', default="1900-01-01",
                    help='Date of birth for enrolling new ID.')
args = parser.parse_args()

def req(IP, mode, image):
    # URLs for different posts
    urls = {
        "detector": "http://{}:5000/api/v1/face_detector".format(IP),
        "identify": "http://{}:5000/api/v1/identify".format(IP),
        "liveness": "http://{}:5000//api/v1/liveness".format(IP),
    }
    # Convert image to file stream
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # Send Request
    try:
        r = requests.post(urls[mode], files={"img" : buffered.getvalue()})
        return json.loads(r.text)
    except Exception as e:
        return {'success': False}

def decode_mtcnn_message(message):
    face_detected = message['face_detected']
    b_box = message['bounding_box']
    distance = message['distance']
    return face_detected, b_box, distance

def decode_facenet_message(message):
    f_vect_shape = message['f_vect_shape']
    ID = message['ID']
    return f_vect_shape, ID


# Box colors in BGR 
color_far = (255, 0, 0) 
color_near = (0, 0, 255) 
color_ok = (0, 255, 0) 
color_error = (255, 0, 255) 
  
# Line thickness of 2 px 
thickness = 2

# define a video capture object 
vid = cv2.VideoCapture(0) 


while(True): 
      
    # Capture the video frame by frame 
    ret, frame = vid.read() 

    # Convert to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    img = Image.fromarray(img)

    # Detect face
    response = req(args.IP, "detector", img)
    if response['success']:
        face_detected, b_box, distance = decode_mtcnn_message(response["message"])
        
        if face_detected:
            x0, y0, x1, y1 = b_box

            # Check distance
            if distance == "too_far":
                color = color_far
            elif distance == "ok":
                color = color_ok
            elif distance == "too_close":
                color = color_near
            else:
                color = color_error

            # Draw rectangle
            frame = cv2.rectangle(frame, (x0,y0), (x1,y1), color, thickness)

    # Display the resulting frame 
    cv2.imshow('frame', frame) 
        
    # The 'q' button is set as the quitting button you may use any 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    # Also break if distance is ok
    if distance == "ok":
        cv2.waitKey(100) 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

# Crop face image
img = img.crop(b_box) 

# Get feature vector
response = req(args.IP, "identify", img)
print(response)
'''
if response['success']:
    f_vect_shape, ID = decode_facenet_message(response["message"])
    print(f_vect_shape)
    print(ID)
'''