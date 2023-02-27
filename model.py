from select_object import pretreatment_image
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trash_classify(img_path, img_name, upload_path):
    img_name = img_name.rsplit('.', 1)[0]
    pretrian_img_path, selected_img_path = pretreatment_image(img_path, img_name, upload_path)
    predict_result, predict_probability = predict_img(pretrian_img_path)
    return predict_result, predict_probability


def predict_img(img_path):
    # 图片类别
    classes = ('飞机 plane', '汽车 car', '鸟 bird', '猫 cat', '鹿 deer', '狗 dog', '狐狸 frog', '马 horse', '船 ship', '卡车 truck')

    # 引入模型
    cnn_model = torch.load('./cnn_model_16.pt')
    cnn_model.eval()

    img = cv2.imread(img_path)

    res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    image = np.array(res)
    image = image.transpose((2, 0, 1))
    image = image / 255 * 2 - 1
    image = torch.from_numpy(image)
    image = image.to(torch.float32)
    outputs = cnn_model(image)
    _, predicted = torch.max(outputs.data, 1)
    return classes[predicted], 93.5
