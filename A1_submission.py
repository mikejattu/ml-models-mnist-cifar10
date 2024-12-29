"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

import torch

import torch.nn as nn
import torch.nn.functional as F

import random
import torchvision

from tqdm import tqdm

from torch.utils.data import random_split

random_seed = 88
torch.manual_seed(random_seed)

def logistic_regression(device):
    '''
    Description: This function is used to implement the logistic regression model
    Input: device (GPU or CPU)
    Output: results (Model)
    '''
    # TODO: implement logistic regression here
    n_epochs = 10
    batch_size_train = 128
    batch_size_test = 1000
    learning_rate = 1e-2
    lambdaL = 0.001
    momentum = 0.95

    torch.backends.cudnn.enabled = False

    results = dict(
        model= LogisticRegression().to(device)
    )   
    # Setting the Optimizer
    optimizer = torch.optim.SGD(results["model"].parameters(), lr= learning_rate, momentum = momentum, weight_decay = lambdaL)

    # Setting the Loss function
    loss = nn.CrossEntropyLoss()

    # Getting the training, Validation and Test Sets
    trainSet, valSet, testSet = getTrainValSets()

    # Getting the dataloaders for Training, Validation and Test Sets
    train_loader = torch.utils.data.DataLoader(trainSet,batch_size= batch_size_train, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(valSet,batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(testSet,batch_size=batch_size_test, shuffle=True)

    # Training the Model
    trainModel(results["model"],train_loader,validation_loader,n_epochs,optimizer,loss,device)

    return results

def trainModel(model,train_loader,validation_loader,n_epochs,optimizer,lossFunc,device):
    """
    Description: This function is used to train the model
    Input: model (Logistic Regression), train_loader (Training Loader), validation_loader (Validation Loader), n_epochs (Number of Epochs), optimizer (Optimizer), lossFunc (Loss Function), device (GPU or CPU)
    Output: None
    """
    for epoch in range(n_epochs):
        # Iterating through each (input,target) pair in the train_loader
        for inputs, targets in train_loader:
            # Reset gradients before backpropagation to avoid accumulation from previous steps
            optimizer.zero_grad()
            # Forward pass: Compute model predictions (outputs) for the current batch of inputs
            outputs = model(inputs.to(device))
            # Calculate the loss between model's predictions and actual targets
            loss = lossFunc(outputs.to(device),targets.to(device))
            # Backward pass: Compute gradients based on the loss
            loss.backward()
            # Update model parameters based on gradients and the optimization algorithm 
            optimizer.step()
            # Performing validation testing
        if random.randint(1, 10)%2 == 0:
            accuracy,val_loss = validation_testing(model,lossFunc,validation_loader,device)
            # Print validation metrics
            print('\n')
            print(f'Epoch {epoch+1}/{n_epochs}, Accuracy: {accuracy}')

def validation_testing(model,lossFunc,val_loader,device):
    """
    Description: This function is used to perform validation testing on the model.
    Inputs: 
        model (Logistic Regression): The model to be evaluated.
        lossFunc (Loss Function): The loss function to compute validation loss.
        val_loader (Validation Loader): DataLoader for validation data.
        device (GPU or CPU): The device on which the model and data are located.
    Outputs: 
        accuracy (float): Accuracy of the model on the validation set.
        val_loss (float): Validation loss.
    """
    # Set the model to evaluation mode
    model.eval()
    val_loss = 0 # Initialize the validation loss
    correct = 0 # Initialize the count of correct predictions
    # Disable gradient calculation for validation 
    with torch.no_grad():
        # Loop through validation dataset (input, target) pairs
        for inputs, targets in val_loader:
            # Move inputs to the correct device (e.g., GPU or CPU) and compute model outputs
            outputs = model(inputs.to(device))
            # Compute the loss for the current batch
            val_loss = lossFunc(outputs.to(device), targets.to(device))
            # Get the predicted labels by finding the index of the maximum logit (for classification)
            pred = outputs.data.max(1, keepdim=False)[1].to(device)
            # Update the count of correct predictions by comparing with actual targets
            correct += pred.eq(targets.to(device)).sum()

    # Calculate accuracy as the percentage of correct predictions
    accuracy = 100. * correct / len(val_loader.dataset)

    return accuracy,val_loss

def getTrainValSets():
    """
    Description: This function is used to get the training and validation sets
    Input: None
    Outputs: 
        - MNIST_training_set (Training Set)
        - MNIST_validation_set (Validation Set)
        - MNIST_test_set (Test Set)"""
    MNIST_training = torchvision.datasets.MNIST('.', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    MNIST_test_set = torchvision.datasets.MNIST('.', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    val_indices = list(range(len(MNIST_training) - 12000 , len(MNIST_training)))
    train_indices = list(range(len(MNIST_training) - 12000))
    # create a training and a validation set
    MNIST_validation_set = torch.utils.data.Subset(MNIST_training, val_indices)
    MNIST_training_set = torch.utils.data.Subset(MNIST_training, train_indices)

    return MNIST_training_set,MNIST_validation_set,MNIST_test_set

class FNN(nn.Module):
    """
    Description: This class is used to define the Fully-connected Neural Network
    """
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.flatten = nn.Flatten()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.lossFuction = nn.CrossEntropyLoss()
        self.layerStack = nn.Sequential(
            nn.Linear(32*32*3, 64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        # Flatten input images for the neural network
        x = self.flatten(x)
        return self.layerStack(x)

    def get_loss(self, output, target):
        return self.lossFuction(output,target)

class LogisticRegression(nn.Module):
    """
    Description: This class is used to define the Logistic Regression Model
    """
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.softmax(self.fc(x), dim = 1)
        return x

def tune_hyper_parameter(target_metric, device):
    """
    Description: This function performs hyperparameter tuning for Logistic Regression and FNN.
    Input: target_metric (performance metric), device (CPU or GPU)
    Output: best_params (best hyperparameters), best_metric (best performance metrics)
    """
    # TODO: implement logistic regression and FNN hyper-parameter tuning here
    best_params = [
                    {
                    "logistic_regression":
                        {
                            "Learning Rate": None,
                            "Lambda": None,
                        }
                    },
                    {
                    "FNN":
                        {
                        "Learning Rate": None,
                        "Lambda": None,
                        }
                    }
]
    best_metric = [
                    {
                    "logistic_regression":
                        {
                        "accuracy":None
                        }},
                    {"FNN":
                        {
                        "accuracy":None
                        }
                    }
]
        # Define training parameters
    batch_size_train = 128
    n_epochs = 4
    best_params_LR= None
    best_metric_LR = 0
    best_params_FNN= None
    best_metric_FNN = 0
    
    # Define hyperparameter search space for Logistic Regression
    params_grid = {
        'Learning Rate': [1e-3,1e-5],
        'Lambda': [0.0000001,0.000008,0.0002,0.0001]
        
    }
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the loss function
    loss = nn.CrossEntropyLoss()

    # Get training and validation sets for MNIST
    trainSet, valSet, testSet = getTrainValSets()
    train_loader = torch.utils.data.DataLoader(trainSet,batch_size= batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valSet,batch_size=batch_size_train, shuffle=True)

    # Perform hyperparameter tuning for Logistic Regression
    total_iterations_LR = (len(params_grid['Learning Rate']) * len(params_grid['Lambda']))
    pbar = tqdm(total=total_iterations_LR, desc='Tuning Logistic Regression', ncols=100)
    model= LogisticRegression().to(device)
    initial_state_dict = model.state_dict() #  model’s parameters in its initial state, before any training or optimization has occurred.
    
    for learning_rate in params_grid['Learning Rate']:
        for lambdaL in params_grid['Lambda']:
            # reseting the model’s parameters back to the original, the untrained state 
            model.load_state_dict(initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay = lambdaL)
            trainModel(model,train_loader,validation_loader,n_epochs,optimizer,loss,device)
            accuracy,val_loss = validation_testing(model,loss,validation_loader,device)
            if accuracy.item() > best_metric_LR:
                best_params_LR = {'Learning Rate': learning_rate, 'Lambda': lambdaL}
                best_metric_LR = accuracy.item()
            pbar.update(1)  # Update the progress bar

    pbar.close()
    best_params[0]["logistic_regression"] = best_params_LR
    best_metric[0]["logistic_regression"]["accuracy"] = best_metric_LR

    # Hyperparameter tuning for FNN
    params_grid = {
        'Learning Rate': [1e-3,1e-5],
        'Lambda': [0.0000001,0.000008,0.0002,0.0001]
        
    }
    loss = nn.CrossEntropyLoss()
    train_loader,validation_loader = get_dataloadersFNN(batch_size_train)

    # Perform hyperparameter tuning for FNN (same logic as above for Logistic Regression)
    total_iterations_FNN = (len(params_grid['Learning Rate']) *
                            len(params_grid['Lambda']))
    pbar = tqdm(total=total_iterations_FNN, desc='Tuning FNN', ncols=100)
    model= FNN('ce', 10).to(device)

    #  model’s parameters in its initial state, before any training or optimization has occurred.
    initial_state_dict = model.state_dict() 

    for learning_rate in params_grid['Learning Rate']:
        for lambdaL in params_grid['Lambda']:
            # reseting the model’s parameters back to the original, the untrained state 
            model.load_state_dict(initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay = lambdaL)
            trainFNN(model, optimizer, train_loader, device)
            accuracy = validationFNN(model,validation_loader,device)
            if accuracy.item() > best_metric_FNN:
                best_params_FNN = {'Learning Rate': learning_rate, 'Lambda': lambdaL}
                best_metric_FNN = accuracy.item()
            pbar.update(1)  # Update the progress bar
    pbar.close()
    best_params[1]["FNN"] = best_params_FNN
    best_metric[1]["FNN"]["accuracy"] = best_metric_FNN

    return best_params, best_metric

def trainFNN(net, optimizer, train_loader, device):
    """
    Description: This function trains the Fully-connected Neural Network (FNN) model.
                    This Function is taken from FNN_main.py
    """
    net.train()
    avg_loss = 0
    for epoch in range(8):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = net.get_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_sc = loss.item()

            avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

def get_dataloadersFNN(batch_size):
    """

    Description: This function retrieves the CIFAR-10 training and validation data loaders.
                    This function is taken from FNN_main.py
    """

    CIFAR_training = torchvision.datasets.CIFAR10('.', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    CIFAR_test_set = torchvision.datasets.CIFAR10('.', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # create a training and a validation set
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size = batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(CIFAR_val_set, batch_size = batch_size, shuffle=True)
    return train_loader, validation_loader

def validationFNN(net, validation_loader, device):
    """
    Description: This function evaluates the performance of the Fully-connected Neural Network (FNN) on a validation dataset.
                It computes the average validation loss and the accuracy of the model.
                This function is taken from FNN_main.py

    """
    net.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = net.get_loss(output, target)
        validation_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    validation_loss /= len(validation_loader.dataset)
    #print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #    validation_loss, correct, len(validation_loader.dataset),
    #    100. * correct / len(validation_loader.dataset)))
    return 100* correct / len(validation_loader.dataset)