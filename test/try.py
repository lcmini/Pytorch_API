from  PIL import  Image
from torchvision import transforms
import torch
import  numpy as np


path = "data/train/ants/6743948_2b8c096dda.jpg"
pic = Image.open(path)
pic.show()
# print(pic)
pic = np.array(pic)
print(pic)
trans_tensor = transforms.ToTensor()
out = trans_tensor(pic)
print(out)