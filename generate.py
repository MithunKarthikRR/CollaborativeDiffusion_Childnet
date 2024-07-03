import torch
from torch.backends import cudnn
import cv2

from parameters import *
from trainer import Trainer
from tester import Tester
from data_loader import CustomDataLoader
from utils import make_folder
from augmentations import *
import os.path as osp
import torch
import timeit
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
# from torchvision import transforms
import torchvision.transforms as transforms

from networks import get_model
from utils import *
# from PIL import Image
import time

from metrics import SegMetric
from time import  time


facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                        'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring',
                        'necklace',
                        'neck', 'cloth']


cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
torch.cuda.manual_seed(2020)


def load_model(model_path):
    model_name="FaceParseNet50"
    model= get_model(model_name, n_classes=19, pretrained=False).cuda()

    torch_saved_obj_path=(model_path)
    torch_saved_obj= torch.load(torch_saved_obj_path )
    model.load_state_dict(torch_saved_obj)
    model.eval()
    print("Model loaded")
    return model

def preprocess_img(img):
    transform = transforms.ToTensor()
    resizer = transforms.Resize(512)
    tensor_img = transform(img)
    tensor_img=resizer(tensor_img)

    tensor_img=torch.unsqueeze(tensor_img, 0)

    assert tensor_img.shape[0] == 1
    assert tensor_img.shape[1] == 3
    assert tensor_img.shape[2] == 512
    assert tensor_img.shape[3] == 512
    print("Preprocessing done")
    return tensor_img

def face_parser(img, model_name,model_path, color_results):

    model=load_model(model_path)
    img=preprocess_img(img)
    img = img.cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        outputs = model(img)
        if model_name == "FaceParseNet":
            outputs = outputs[0][-1]
        h, w = 512, 512

        outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
        pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
    print("Face Parsing done")
    if color_results: 
        imsize=512
        labels_predict_plain = generate_label_plain(outputs, imsize)
        r=labels_predict_plain[0]
        filename = os.path.basename(img_path).split('.')[0]
        output_path = f"/content/drive/MyDrive/mask/FaceParsing.PyTorch/result/{filename}.png"
        
        cv2.imwrite(output_path, r)
    print("File added in FaceParsing.PyTorch/result/")


    return outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Parsing Model")
    parser.add_argument("--src_path", type=str, help="Path to the input image", required=True)
    args = parser.parse_args()

    img_path = args.src_path
    img = cv2.imread(img_path)
    model_name = "FaceParseNet"
    model_path = "./models/FaceParseNet50/38_G.pth"
    face_parser(img, model_name, model_path, color_results=True)

