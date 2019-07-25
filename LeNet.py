import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from torchvision import datasets
from PIL import Image
import os

# torch.set_default_dtype(torch.uint8)


def quantize_arr(arr):
    """Quantization based on linear rescaling over min/max range.
    """
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val > 0:
        quantized = np.round(255 * (arr - min_val) / (max_val - min_val))
    else:
        quantized = np.zeros(arr.shape)
    quantized = quantized.astype(np.uint8)
    min_val = min_val.astype(np.float32)
    max_val = max_val.astype(np.float32)
    return quantized, min_val, max_val


def print_weights(model, weights_path):

    # Load Weights
    model.load_weights(weights_path)

    for layer in model.layers:
        print(layer.get_weights())


def write_file(output_path, data, write=False):

    path = output_path.split("/")
    if len(path) == 2:
        if not os.path.isdir(path[0]):
            os.mkdir(path[0])
    if write:
        with open(output_path, 'w') as f:
            f.write(data)
    else:
        with open(output_path, 'a') as f:
            f.write(data)


def Quantize_weights(weights_float):

    weights_uint8 = []
    layers = []

    for weight_float in weights_float:
        for a in weight_float:
            weight, min_val, max_val = quantize_arr(a)
            weights_uint8.append(weight)
    layers.append(weights)


def data_processing(num_workers=0, batch_size=32, valid_sample=0.2):

    transform = [transforms.Pad(2), transforms.ToTensor()]

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
        # 32 x 32 x 1
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=0, stride=1)
        # 28 x 28 x 6
        self.pool1 = nn.AvgPool2d((2, 2), stride=2)
        # 14 x 14 x 6
        self.conv2 = nn.Conv2d(6, 16, (5, 5), padding=0, stride=1)
        # 10 x 10 x 16
        self.pool2 = nn.AvgPool2d((2, 2), stride=2)
        # 5 x 5 x 16
        self.conv3 = nn.Conv2d(16, 120, (5, 5), padding=0, stride=1)
        # 1 x 1 x 120
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # Choose either view or flatten (as you like)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


def Train(model,
          train_loader,
          validation_loader,
          optimizer,
          criterion,
          n_epochs=40):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

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
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.to(device))
            # calculate the loss
            loss = criterion(output, target.to(device))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in validation_loader:
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
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


def Visualize(model, input_img):
    param = list(model.parameters())

    data = model.fc2.forward(model.fc1.forward(
        torch.flatten(model.conv3.forward(
            model.pool2.forward(
                model.conv2.forward(
                    model.pool1.forward(model.conv1.forward(input_img))))),
                        start_dim=1)))
    write_file("fc2/output.txt", "{}".format(data.data), write=True)
    for i in range(param[8].shape[0]):
        write_file("fc2/weights.txt", "{}, Bias: {}\n".format(param[8][i].data, param[9][i]))
    # for i in range(param[6].shape[0]):
        # write_file("conv3/filter_weights.txt",
        #            "{}, Bias: {} \n".format(param[4][i].data, param[5][i]))
        # write_file("conv3/filter_weights.txt", "{}\n".format(img))

        # write_file("pool2/Filter" + str(i) + ".txt",
        #            "{}".format(img),
        #            write=True)
        # img = Image.fromarray(img)
        # img.save("pool2/Filter" + str(i) + ".png")


def Similarity(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    result = np.array(img1) - np.array(img2)
    print("{}%".format((1 - np.average(result) / 255) * 100))


if __name__ == "__main__":

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    #  validation sample
    valid_sample = 0.2

    # initilize the model
    model = LeNet().float()

    ##### Testing
    # test = np.random.randn(1,32,32)
    # print(test.shape)
    # result = model(torch.from_numpy(test).unsqueeze(0).float())
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
    # Test(model, classes, test_loader)

    # Visualize weights and feature maps
    Visualize(model, next(iter(test_loader))[0][0].unsqueeze(0))

    # Similarity("Filter0.png", "Filter3.png")
    # for param in model.parameters():
    #     print(param.shape)