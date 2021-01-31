import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Training import ShieldKeras

import warnings

warnings.filterwarnings('ignore')
class MyDataset(Dataset):
    def __init__(self, trainX, trainy):
        self.trainX = trainX
        self.trainy = trainy

    def __len__(self):
        return self.trainX.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.trainX[idx]), torch.tensor(self.trainy[idx]).unsqueeze(-1)


def f():

    model = nn.Linear(in_features=11, out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    trainX = torch.randn(12000, 11)
    trainY = torch.randn(12000)
    dataset = MyDataset(trainX, trainY)
    Model = ShieldKeras(model)

    Model.compile(optimizer=optimizer, loss=criterion)
    Model.fit(train_dataset=dataset, validation_dataset=dataset, batch_size=32,epochs=10)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
