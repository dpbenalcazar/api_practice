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
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval() 
  
def mtcnn_detection(pil_image): 
    # Get image size
    W, H = pil_image.size

    try:
        # Detect face
        box_mtcnn = mtcnn.detect(pil_image)[0][0]
        x0, y0, x1, y1 = box_mtcnn
        box_pil = [int(x0), int(y0), int(x1), int(y1)]

        # Check distance
        dy = y1 - y0
        if dy/H < 0.4:
            distance = "too_far"
        elif dy/H >= 0.4 and dy/H < 0.5:
            distance = "ok"
        else:
            distance = "too_close"

    except:
        box_pil = None
        distance = None

    return box_pil, distance

def facenet_embedding(face_image):
    # Convert to tensor
    face_tensor = tform(face_image).to(device)

    # Get feature vector
    embedding = resnet(face_tensor.unsqueeze(0))

    return embedding.clone().detach().cpu().numpy()
    
