import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, ReLU
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(self.path)  # a list type

    def __getitem__(self, item):
        img_name = self.image_list[item]  # list[0] [1] [2],get name as '258217966_d9d90d18d3.jpg'
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # get a  picture path in tran data
        # img = cv.imread(img_item_path)
        img = Image.open(img_item_path).convert('RGB')

        # Transforms
        trans_resize = transforms.Resize((128, 128))
        img = trans_resize(img)
        trans_totensor = transforms.ToTensor()
        img = trans_totensor(img)

        label = self.label_dir
        if label == "ants":
            label = 1
        else:
            label = 2
        return img, label

    def __len__(self):
        return len(self.image_list)  # list path


# 构建神经网络
class Net(nn.Module):  # 定义网络模块
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=5, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=18, out_channels=27, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, padding=0, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(27 * 3 * 3, 9),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# device set
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# path set
train_root_dir = "data/train"
test_root_dir = "data/val"
ant_dir = "ants"
bee_dir = "bees"

writer = SummaryWriter("logs")

# train dataset
train_ant_dataset = Mydata(train_root_dir, ant_dir)
train_bee_dataset = Mydata(train_root_dir, bee_dir)
train_dataset = train_ant_dataset + train_bee_dataset
# test dataset
test_ant_dataset = Mydata(test_root_dir, ant_dir)
test_bee_dataset = Mydata(test_root_dir, bee_dir)
test_dataset = test_ant_dataset + test_bee_dataset

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

print("tran_dataset 的长度是" + str(len(train_dataset)))
print("test_dataset 的长度是" + str(len(test_dataset)))

net = Net()
print(net)
step = 0
for data in train_dataloader:
    imgs, targets = data
    print(imgs)
    print(targets)
    print('\n')
    print(imgs.shape)
    output = net(imgs)
    # print(output)
    print(output.shape)
    writer.add_images("intput", imgs, step)
    output = torch.reshape(output, (-1, 3, 12, 8))
    writer.add_images("output", output, step)
    step += 1
writer.close()
# torch.Size(bachsize,channel,h,w)
# step = 0
# for data in train_dataloader:
#     imgs, targets = data
#     print(imgs.shape)
# print(targets)
# writer.add_images("train_dataloader", imgs, step)
# step += 1
# writer.close()
'''''''''''
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)
print(kernel.shape)
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input,kernel,stride=1,padding=1)
print(output)
'''''''''
