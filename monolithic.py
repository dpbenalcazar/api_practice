# import the opencv library 
import cv2 
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('../facenet-pytorch/')
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
tform = transforms.Compose([transforms.Resize([160, 160]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

# Create a face detection pipeline using MTCNN:
mtcnn = MTCNN(select_largest=False, device=device)

# Create an inception resnet (in eval mode):
#resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Box colors in BGR 
color_far = (255, 0, 0) 
color_near = (0, 0, 255) 
color_ok = (0, 255, 0) 
  
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
    W, H = img.size

    try:
        # Detect face
        box_mtcnn = mtcnn.detect(img)[0][0]
        x0, y0, x1, y1 = box_mtcnn.astype(int)
        box_pil = (x0, y0, x1, y1)

        # Check distance
        dy = y1 - y0
        if dy/H < 0.4:
            color = color_far
        elif dy/H >= 0.4 and dy/H < 0.5:
            color = color_ok
        else:
            color = color_near

        # Draw rectangle
        frame = cv2.rectangle(frame, (x0,y0), (x1,y1), color, thickness)

        # Crop face
        #img = img.crop(box_pil)
        #img_tensor = tform(img).to(device)

        # Get feature vector
        #embedding = resnet(img_tensor.unsqueeze(0))
        #print(embedding.shape)

    except:
        pass

    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # The 'q' button is set as the quitting button you may use any 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

