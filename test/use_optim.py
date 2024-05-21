import os
import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


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
            label = 0
        else:
            label = 1
        return img, label

    def __len__(self):
        return len(self.image_list)  # list path


# 构建神经网络
class Net(nn.Module):  # 定义网络模块
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16384, 2)
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

step = 0

# 初始化神经网络
net = Net()

# loss函数
loss = torch.nn.CrossEntropyLoss()

# 优化器(input weights,learning rate)
optim = torch.optim.SGD(net.parameters(), lr=0.012)
for epoch in range(20):
    running_loss = 0.0
    for data in train_dataloader:
        input_img, targets = data
        output = net(input_img)
        # 使用Loss检测误差
        result_loss = loss(output, targets)
        # 优化器梯度清0
        optim.zero_grad()
        # 反向传递，计算梯度
        result_loss.backward()
        # 利用梯度进行调优
        optim.step()
        running_loss = running_loss + result_loss
        # print(result_loss)
    print(running_loss)
    # print(input_img.shape)
    # print(targets)
    # writer.add_images("train_dataloader", input_img, step)
    # step += 1
writer.close()
