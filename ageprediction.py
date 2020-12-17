import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_excel('議員一覧表.xlsx')
print(df)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, t_data, transform=transforms.ToTensor()):
        self.x_data = np.array(x_data).astype('float32') / 255
        self.t_data = t_data / 100
        self.transform = transform
        self.x_data_pil = []
        for image in x_data:
            self.x_data_pil.append(Image.fromarray(np.uint8(image)))

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.x_data_pil[idx]), torch.tensor(
            self.t_data[idx], dtype=torch.float)


# ファイル読み込み
image_list = []
for i in tqdm(range(1, df.shape[0] + 1)):
    image_list.append(np.array(Image.open('img2/face{}.jpg'.format(i))))

# データセット作成
dataset = MyDataset(image_list, df['年齢'])
# dataset = MyDataset(image_list, df['性別(男0,女1)'])

# 学習データと検証データに分割
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
dataset_train, dataset_valid = torch.utils.data.random_split(
    dataset, [train_size, val_size])

# データローダー作成
BATCH_SIZE = 10
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        channel = channel_out // 4

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm2d(channel_out)

        self.shortcut = self._shortcut(channel_in, channel_out)

    def forward(self, x):
        h = self.conv1(x)
        h = self.batchnorm1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.batchnorm2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.batchnorm3(h)
        shortcut = self.shortcut(x)
        y = self.relu(h + shortcut)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.block1 = Block(1, 16)
        self.block2 = Block(16, 16)

        self.fc1 = nn.Linear(16 * 9 * 7, 480)
        self.fc2 = nn.Linear(480, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 60)
        self.fc5 = nn.Linear(60, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)
        x = self.sigmoid(x)
        x = torch.flatten(x)
        return x


conv_net = Net()
conv_net.to(device)

n_epochs = 50
lr = 0.001

optimizer = optim.Adam(conv_net.parameters(), lr=lr)
loss_function = nn.MSELoss()

log_loss_train = []
log_loss_valid = []
log_accuracy_train = []
log_accuracy_valid = []
start = time.time()
for epoch in range(n_epochs):
    conv_net.train()
    losses_train = []
    for x, t in dataloader_train:
        conv_net.zero_grad()            # 勾配の初期化
        x = x.to(device)                # テンソルをGPUに移動
        t = t.to(device)
        y = conv_net.forward(x)         # 順伝播
        loss = loss_function(y, t)      # 誤差関数の計算
        loss.backward()                 # 誤差の逆伝播
        optimizer.step()                # パラメータの更新
        losses_train.append(loss.tolist())
    loss_train = np.mean(losses_train)
    log_loss_train.append(loss_train)
    accuracy_train = np.sqrt(loss_train) * 100
    log_accuracy_train.append(accuracy_train)

    conv_net.eval()
    losses_valid = []
    for x, t in dataloader_valid:
        x = x.to(device)                # テンソルをGPUに移動
        t = t.to(device)
        y = conv_net.forward(x)         # 順伝播
        loss = loss_function(y, t)      # 誤差関数の計算
        losses_valid.append(loss.tolist())
    loss_valid = np.mean(losses_valid)
    log_loss_valid.append(loss_valid)
    accuracy_valid = np.sqrt(loss_valid) * 100
    log_accuracy_valid.append(accuracy_valid)

    print('[{:.3f}] EPOCH: {}, Train [Loss: {:}, Accuracy: {:.3f}], Valid [Loss: {:}, Accuracy: {:.3f}]'.format(
        time.time() - start,
        epoch,
        loss_train,
        accuracy_train,
        loss_valid,
        accuracy_valid
    ))

plt.figure()
plt.title('train: {}, vaild: {}'.format(
    log_loss_train[-1], log_loss_valid[-1]))
plt.plot(log_loss_train, label='train')
plt.plot(log_loss_valid, label='vaild')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss.png')

plt.figure()
plt.title('train: {:.2f}, vaild: {:.2f}'.format(
    log_accuracy_train[-1], log_accuracy_valid[-1]))
plt.plot(log_accuracy_train, label='train')
plt.plot(log_accuracy_valid, label='vaild')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
