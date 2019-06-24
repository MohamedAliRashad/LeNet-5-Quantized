import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchsummary import summary

# torch.set_default_dtype(torch.uint8)

def data_processing(num_workers=0, batch_size=32, valid_sample=0.2):

    transform = [transforms.ToTensor()]

    # choose the training and test datasets
    train_data = datasets.MNIST(root='data',
                                train=True,
                                download=True,
                                transform=transforms.Compose(transform))
    test_data = datasets.MNIST(root='data',
                               train=False,
                               download=True,
                               transform=transforms.Compose(transform))

    # Creating validation sampler
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(valid_sample * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define sampler for batches
    trainSampler = SubsetRandomSampler(train_idx)
    validationSampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              sampler=trainSampler,
                              num_workers=num_workers)
    validation_loader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   sampler=validationSampler,
                                   num_workers=num_workers)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             num_workers=num_workers)

    return train_loader, validation_loader, test_loader


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 28 x 28 x 1
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        # 28 x 28 x 6
        self.pool1 = nn.AvgPool2d((2,2), stride=1)
        # 27 x 27 x 6
        self.conv2 = nn.Conv2d(6, 16, (5, 5), padding=2)
        # 14 x 14 x 16
        self.conv3 = nn.Conv2d(16, 120, (5, 5), padding=2)
        # 6 x 6 x 120
        self.pool2 = nn.AvgPool2d((2,2), stride=2)
        self.fc1 = nn.Linear(6 * 6 * 120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = self.pool2(x)
        # Choose either view or flatten (as you like)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


def Train(model,
          train_loader,
          validation_loader,
          optimizer,
          criterion,
          n_epochs=40):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model.to(device)

    valid_loss_min = np.Inf

    for epoch in range(n_epochs):

        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_loader:
            # data.to(device)
            # target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in validation_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(validation_loader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.
              format(epoch + 1, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss


def Test(model, classes, test_loader):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()  # prep model for *evaluation*

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
                  (str(i), 100 * class_correct[i] / class_total[i],
                   np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %
          (100. * np.sum(class_correct) / np.sum(class_total),
           np.sum(class_correct), np.sum(class_total)))


if __name__ == "__main__":

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    #  validation sample
    valid_sample = 0.2

    # initilize the model
    model = LeNet()
    
    # print the model shapes in every layer
    # summary(model.to("cuda"), (1, 28, 28))
    
    # classes of MNIST
    classes = list(range(10))

    # specify loss function
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 
    train_loader, validation_loader, test_loader = data_processing(
        num_workers, batch_size, valid_sample)

    # Training the model on GPU if avialable
    # Train(model, train_loader, validation_loader, optimizer, criterion)

    # Loading the model's weights
    model.load_state_dict(torch.load("model.pt"))

    # Testing the model
    Test(model, classes, test_loader)
