from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import mnist

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = nn.Sequential(
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100,10),
        nn.LogSoftmax(dim=1)
)
net.float()

def traino(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#        correct.extend() = pred.eq(target.view_as(pred))
        #print(np.asarray(np.sum(correct)))
        return loss, correct

def train(args, model, device, train_data, optimizer, epoch):
    model.train()
    data = torch.from_numpy(train_data[0].astype('float32')) #.unsqueeze(0)
    target = torch.from_numpy(train_data[1].astype('int64')) #.unsqueeze(0)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred))
    return loss, correct
            
#def test(args, model, device, test_loader):
    #model.eval()
    #test_loss = 0
    #correct = 0
    #with torch.no_grad():
        #for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            #output = model(data)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))

def main(set_lr ,set_r_seed):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST easy ex change w lr')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    
    args.epochs=1
    args.lr = set_lr
    args.seed = set_r_seed
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    x,t = mnist.load_1000()
    #data = [x.reshape(100,10, 28,28),t.reshape(100,10, -1)]
    data = [x,t]
    
#    model = Net().to(device)
    model = net
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ress = []
    for epoch in range(1, args.epochs + 1):
        res = train(args, model, device, data, optimizer, epoch)
        #test(args, model, device, test_loader)
        ress.append(res)
#        print( epoch, np.sum(np.asarray(res[1])) )
    return ress[0] 

if __name__ == '__main__':
    import numpy as np
    learning_rates = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    random_seeds = np.random.randint(low=1, high=3000, size=5)
    corr_results = [] #dict.fromkeys(learning_rates, dict.fromkeys(random_seeds, []))
    loss_results = [] #dict.fromkeys(learning_rates, dict.fromkeys(random_seeds, []))
    for i,l in enumerate(learning_rates):
        corr_results.append([])
        loss_results.append([])
        for j,r in enumerate(random_seeds):
           corr_results[i].append([])
           loss_results[i].append([])
           loss, corr = main(l, r)
           print (l, r, loss, np.sum(np.asarray(corr)))
           corr_results[i][j] = corr
           loss_results[i][j] = loss

    import pickle
    with open ('corr_res', 'wb') as f:
        pickle.dump(corr_results, f)

    with open ('loss_res', 'wb') as f:
        pickle.dump(loss_results, f)

