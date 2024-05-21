import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter('logs')
path = "data/train/ants/6743948_2b8c096dda.jpg"
img = cv.imread(path, 1)
img2 = Image.open(path)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)
writer.close()

# Normalize(归一化)
# put in tensor picture
print(img_tensor[1][1][1])
trans_Normalize = transforms.Normalize([2, 0.5, 1.5], [2, 1, 0.5])
img_Normalize = trans_Normalize(img_tensor)
writer.add_image("Normalize", img_Normalize)
print(img_Normalize[1][1][1])
writer.close()

# Resize
# 读取的是PILImage 图片，不能是imread读取
print(img2.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img2)
img_resize = trans_totensor(img_resize)

writer.add_image("resize", img_resize,1)
print(img_resize.size)

# Compose
trans_resize2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize2,trans_totensor])
img_resize2 = trans_compose(img2)
writer.add_image("resize", img_resize2,2)
writer.close()

