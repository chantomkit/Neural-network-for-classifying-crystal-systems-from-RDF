import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

filename = "systemClassificationByRDF_model.pt"

# This determines what the program will execute.
# "train" mode is to train the neural network from zero, which will take a lot of time.
# "testAll" mode is to let the neural network goes through all the data in the testset, giving the overall performance.
# "testOne" mode is to pass one random data to the neural network and also displaying that data on screen.
types = {0: "Triclinic", 1: "Monoclinic", 2: "Orthorhombic", 3: "Tetragonal", 4: "Trigonal",
         5: "Hexagonal", 6: "Cubic"}
mode = {"train": False, "testAll": False, "testOne": False}

# Getting toy datasets (only set download=True in the first run)
samples = np.load("training_data.npy", allow_pickle=True)
np.random.shuffle(samples)
trainPercentage = 0.9
trainSize = int(trainPercentage * len(samples))

# Divide datasets into batches with shuffling
X = torch.Tensor([i[0] for i in samples])
y = torch.Tensor([i[1] for i in samples])

train_X = X[:trainSize]
train_y = y[:trainSize]
test_X = X[trainSize:]
test_y = y[trainSize:]
inputFeatures = len(train_X[0])
outputValues = len(train_y[0])

# print(train_X, train_y)
# for i in range(len(test_X)):
#     print(i)
#     if len(test_X[i]) != 1099 or len(test_y[i]) != 7:
#         print("Some array does not have correct length")
#         break

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(inputFeatures, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, outputValues)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Calling and loading the neural network
net = Net()
# print(net)
net.load_state_dict(torch.load(filename))
net.eval()

class trainNet():
    def train(self, mode):
        if mode.get("train") == True:
            batchSize = 20
            EPOCHS = 15
            decayRate = 0.9
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            for epoch in range(EPOCHS): # 3 full passes over the data
                for i in range(0, len(train_X), batchSize):  # `data` is a batch of data
                    batch_X = train_X[i:i+batchSize]
                    batch_y = train_y[i:i+batchSize]
                    net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                    output = net(batch_X)  # pass in the reshaped batch
                    loss = loss_function(output, batch_y)  # calc and grab the loss value
                    loss.backward()  # apply this loss backwards thru the network's parameters
                    optimizer.step()  # attempt to optimize weights to account for loss/gradients
                lr_scheduler.step()
                print(f"Epoch: {epoch}")
                print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
            torch.save(net.state_dict(), filename)
        else:
            print("Note that train mode is not activated.")

class testNet():
    def check(self, mode):
        if mode.get("testAll") == True and mode.get("testOne") == True:
            print("Note that both testAll and testOne mode are on, it may cause errors.")
            exit()

    def testAll(self, mode):
        if mode.get("testAll") == True:
            correct_train = 0
            total_train = 0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for i in range(len(train_X)):
                    label_train = torch.argmax(train_y[i])
                    output_train = net(train_X[i].view(-1, inputFeatures))
                    # print(output)
                    # output = net(train_X[i].view(-1, 1, 1, inputFeatures))[0]
                    prediction_train = torch.argmax(output_train)
                    # print(prediction)
                    if label_train == prediction_train:
                        correct_train += 1
                    total_train += 1

                for i in range(len(test_X)):
                    label_test = torch.argmax(test_y[i])
                    output_test = net(test_X[i].view(-1, inputFeatures))
                    prediction = torch.argmax(output_test)
                    if label_test == prediction:
                        correct_test += 1
                    total_test += 1
                print("Accuracy for training samples: ", round(correct_train / total_train, 3))
                print("Total training samples: ", total_train)
                print("Accuracy for testing samples: ", round(correct_test / total_test, 3))
                print("Total testing samples: ", total_test)

        else:
            print("Note that testAll mode is not activated.")

    def testOne(self, mode):
        if mode.get("testOne") == True:
            with torch.no_grad():
                for i in range(len(test_X)):
                    label = torch.argmax(test_y[i])
                    output = net(test_X[i].view(-1, inputFeatures))
                    prediction = torch.argmax(output)
                    break
                print(label, prediction)
                print(f"Model Prediction: {types[int(prediction)]}\nLabel: {types[int(label)]}")
        else:
            print("Note that testOne mode is not activated.")

if __name__ == "__main__":
    trainNet.train(Net, mode)
    testNet.check(Net, mode)
    testNet.testAll(Net, mode)
    testNet.testOne(Net, mode)