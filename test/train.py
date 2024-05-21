import os
import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


# use transforms,a til to change picture's size,type,channel

# use the dataset
# first heritage the dataset class
# example:
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
        trans_resize = transforms.Resize((32, 32))
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
            nn.Linear(64*4*4, 2)
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

# tensorboard
# add_scalar("name",value_y,value_x)
# example:
# writer = SummaryWriter("logs")
# for i in range(100):
#    writer.add_scalar("y=2x", 2 * i, i)
# writer.close()

# writer.add_image("name_title",numpy_class_picture/torch_class_picture,step,dataformats='HWC')
# pay attention to dataformats,and step is 1,2,3,4,should change
# example:
# path = "data/train/ants/0013035.jpg"
# img = cv.imread(path,1)
# img = np.array(img)
# print(type(img))
# print(img.shape)
# writer = SummaryWriter("logs")
# writer.add_image("demo or train or test",img,1,dataformats='HWC')
# writer.close()

# use tensorboard --logdir=logs --port=6007 to open tensorboard
# get a new picture,delete webs in logs
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

# 记录训练次数
train_step = 0

# 测试此书
test_step = 0

epoch = 1000

net = Net()

loss = torch.nn.CrossEntropyLoss()

# 优化器(input weights,learning rate)
learning_rate = 0.01
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)
for i in range(epoch):
    print("------第" + str(i + 1) + "轮训练开始-----")
    total_train_loss = 0.0
    for data in train_dataloader:
        input_img, targets = data
        output = net(input_img)
        # 使用Loss检测误差
        train_loss = loss(output, targets)
        # 优化器梯度清0
        optim.zero_grad()
        # 反向传递，计算梯度
        train_loss.backward()
        # 利用梯度进行调优
        optim.step()

        # print("train loss：" + str(result_loss.item()))
        total_train_loss = total_train_loss + train_loss
        train_step += 1
        # print(result_loss)

        writer.add_scalar("train_loss", train_loss, train_step)
    print("every epoch train  loss:" + str(total_train_loss.item()))
    # print(input_img.shape)
    # print(targets)
    # writer.add_images("train_dataloader", input_img, step)
    # step += 1
    # 测试开始
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    with torch.no_grad():
        for data in train_dataloader:
            test_img, targets = data
            outputs = net(test_img)
            test_loss = loss(outputs, targets)
            total_test_loss = test_loss + total_test_loss
            test_step += 1
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy += accuracy

            writer.add_scalar("test_loss", test_loss, test_step)
            writer.add_scalar("test_accuracy", total_test_accuracy / len(test_dataset), test_step)
        print("every epoch test sum loss:" + str(total_test_loss.item()))
        print("整体数据集上的正确率：{}".format(total_test_accuracy / len(test_dataset)))

    torch.save(net, "./weights/torch.pth")
writer.close()
# img,label = Ant_dataset[0],
# get the first picture's label and read this picture
#
# len(Ant_dataset)
# get the length of dataset
