import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default='train.csv')
parser.add_argument('--test_data_dir', type=str, default='testA.csv')
parser.add_argument('--split_ratio', type=float, default=0.9)
parser.add_argument('--dimension', type=int, default=205)
parser.add_argument('--hidden_size', type=int, default=205)
parser.add_argument('--num_of_class', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_known_args()[0]

def DataProcessor(args):
    train_data = pd.read_csv(args.train_data_dir)
    test_data = pd.read_csv(args.test_data_dir)

    train_id, train_x, train_y = train_data['id'], train_data['heartbeat_signals'], train_data['label']
    train_x = torch.tensor([list(map(float, x.split(','))) for x in train_x])
    train_y = torch.nn.functional.one_hot(torch.LongTensor(list(train_y)), num_classes=args.num_of_class)
    train_dataset = TensorDataset(
        train_x[:int(train_x.shape[0]*args.split_ratio)],
        train_y[:int(train_y.shape[0]*args.split_ratio)])
    validate_dataset = TensorDataset(
        train_x[int(train_x.shape[0]*args.split_ratio):],
        train_y[int(train_y.shape[0]*args.split_ratio):])

    test_id, test_x = test_data['id'], test_data['heartbeat_signals']
    test_dataset = torch.tensor([list(map(float, x.split(','))) for x in test_x])

    print('Training Data: %d' %len(train_dataset))
    print('Validate Data: %d' %len(validate_dataset))
    print('Test Data: %d' %len(test_dataset))
    return train_dataset, validate_dataset, test_dataset

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(args.dimension, args.hidden_size,
                            batch_first=True,
                            bidirectional=True,
                            dropout=args.dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * args.hidden_size, args.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.num_of_class))

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        x = self.lstm(x)[0]
        x = torch.max(x, dim=1)[0]
        return self.fc(x)

def validate(model, device, validate_loader, loss_func):
    model.eval()
    total_loss = 0
    for _, (x, y) in enumerate(validate_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
            loss = loss_func(y_pred, y.float())
            total_loss += loss
    return total_loss


def train(model, device, train_loader, optimizer, loss_func):
    model.train()
    total_loss = 0
    for _, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_func(y_pred, y.float())
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss #/ len(train_loader)

def test(model, device, test_dataset):
    model.eval()
    x = test_dataset.to(device)
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.nn.functional.one_hot(torch.argmax(y_pred.cpu(), dim=1), num_classes=4)
    res_csv = pd.DataFrame(y_pred.numpy())
    res_csv.to_csv('result.csv')

if __name__ == '__main__':
    train_dataset, validate_dataset, test_dataset = DataProcessor(args)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    validate_sampler = RandomSampler(validate_dataset)
    validate_loader = DataLoader(validate_dataset, sampler=validate_sampler, batch_size=args.batch_size)

    model = LSTM(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
    for epoch in range(args.epoch):
        training_loss = train(model, device, train_loader, optimizer, loss_func)
        validate_loss = validate(model, device, validate_loader, loss_func)
        print('Epoch %d\tTraining Loss:%.2f\tValidate Loss:%.2f' %(epoch, training_loss, validate_loss))
    test(model, device, test_dataset)



