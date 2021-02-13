# Imports
import torch                                # Importing general library                        
import torchvision                          # Vision based library by Pytorch
import torch.nn as nn                       # Importing all neural network modules,different model architectures like Linear,CNN,loss fns etc.
import torch.optim as optim                 # Importing all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F             # Importing all functions that don't have any parameters like activation fns.
from torch.utils.data import DataLoader     # Gives easier dataset management and creates mini batches to train on etc.It is very useful while training since we tend to manage our dataset a lot
import torchvision.datasets as datasets     # Importing standard datasets easily,not needed when we use custom datasets
import torchvision.transforms as transforms # Importing transformations that we can perform on our dataset

#Create a Neural Network
class NN(nn.Module): #nn.Module is the parent class of our custom class NN
    def __init__(self, input_size, num_classes): #Input_size and no.of classes(output)
        super(NN, self).__init__() #Initialising the parent class,we use super(NN,self)
        #nn.Linear() applies a linear transformation to the incoming data: y=xA^T+b.
        self.fc1 = nn.Linear(input_size, 50) #Linear(input_layer_size,output_layer_size,bias),bias is set to True(default),if false no bias
        self.fc2 = nn.Linear(50, num_classes) #So we have an input layer(size given),a hidden layer(50 neurons) and an output layer(size given)
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) #fc1 is like summation output,called it using self as it belongs to class NN,F is nn.functional library so has relu activation fn
        x = self.fc2(x) #There is no activation fn considered here,however you can consider sigmoid or tanh if wanted 
        return x
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784 #MNIST dataset is 28 * 28 pixels so 28*28 = 784 that is input size.So we flatten our matrix and pass it as a one dimensional array
num_classes = 10 #We have 10 classes [0,9]
learning_rate = 0.001 #Determines the step size at each iteration while moving toward a minimum of a loss function
batch_size = 64 #Batch size to be processed at once
num_epochs = 1 #No.of iterations

# Load Data
train_dataset = datasets.MNIST(root='/home/manju838/coding/env/pytorch_env/Datasets/MNIST/MNIST', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='/home/manju838/coding/env/pytorch_env/Datasets/MNIST/MNIST', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device) #NN class object called model,to(device) is for GPU or CPU

# Loss and optimizer
criterion = nn.CrossEntropyLoss() #This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.The input is expected to contain raw, unnormalized scores for each class.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Get to correct shape
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)