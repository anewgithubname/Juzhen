# %%
import torch, torchvision
# default device is gpu 
device = 'cuda'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#convert the images to flattened tensors
dataset = torchvision.datasets.MNIST(root='/tmp', download=True, transform=torchvision.transforms.ToTensor())
# flatten the input features
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

testdataset = torchvision.datasets.MNIST(root='/tmp', train=False, download=True, transform=torchvision.transforms.ToTensor())
# load all testing images
testloader = torch.utils.data.DataLoader(testdataset, batch_size=10000, shuffle=True)
# load everything into memory
testdata, testtarget = next(iter(testloader))
testdata = testdata.to(device); testtarget = testtarget.to(device)
# %%
# design a MLP neural network model that is 27*27 -> 1024 -> 128 -> 10
# with relu activation function

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # inherit from nn.Module
        super(Net, self).__init__()
        # define the layers
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        # define the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return the output
        return x

# %%
# define the 10-class logistic loss

def logistic_loss(output, target):
    # compute the loss
    loss = nn.CrossEntropyLoss()
    return loss(output, target)

# %%
# time the training process
import time
tic = time.time()

net = Net().to(device)
# define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

iteration = 0
# start training loop with batch size 32
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(loader):
        if(iteration > 10000):
            break
    
        data = data.to(device)
        target = target.to(device)
        
        # zero the gradient
        optimizer.zero_grad()
        # forward pass
        output = net(data)
        # compute the loss
        loss = logistic_loss(output, target)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # print the progress as well as the miscategorization rate on the testing data
        if iteration % 1000 == 0:
            with torch.no_grad():
                testoutput = net(testdata)
                testpred = torch.argmax(testoutput, dim=1)
                testerr = torch.mean((testpred != testtarget).float())
                # print up to 4 decimal points
                print('Iteration: {}, Test error: {:.4f}'.format(iteration, testerr))
                
        iteration = iteration + 1
          
toc = time.time()
print('Time elapsed: {} seconds'.format(toc - tic))
# %%
