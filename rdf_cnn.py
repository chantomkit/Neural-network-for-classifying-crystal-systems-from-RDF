from os import listdir
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

EPOCHS = 30
binmax = 1000
l_rate = 0.001
b_size = 100

# Some naming mechanisms
ratio = np.array([1, 1, 1, 1, 1, 1, 1])
rdf = "td_fullrdf_mp"

if rdf == "td_fullrdf_1l" or "td_fullrdf_al" or "td_fullrdf_al_half":
    rdftitle = "Cluster RDF"
elif rdf == "td_fullrdf_raw":
    rdftitle = "Cluster RDF (No normalization)"
elif rdf == "td_simrdf":
    rdftitle = "PBC RDF"
elif rdf == "td_fullrdf_mp":
    rdftitle = "Cluster RDF (MP)"

name = str(ratio).replace(" ", "").replace("[", "").replace("]", "")
# name = "training_data_big"

# Here import the nn model and dataset sample
modelname = f"model/conv_{rdf}_{name}_{binmax}_{EPOCHS}.pt"
samplename = f"packed_td/{rdf}_{name}_{binmax}_pbc.npy"
# samplename = f"{rdf}/training_data_mp_{binmax}_pbc_big.npy"

types = ["Tri-", "Mono-", "Ortho-", "Tetra-", "Trig-", "Hexa-", "Cubic"]
samples = np.load(samplename, allow_pickle=True)
np.random.shuffle(samples)
X = torch.Tensor([i[0] for i in samples])
y = torch.Tensor([i[1] for i in samples])
spgrp = torch.Tensor([i[2] for i in samples])
# input_channels is defined wrongly, just ignore it and use 1
inC, outF = len(X[0]), len(y[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 8)
        self.conv2 = nn.Conv1d(8, 16, 8)
        self.conv3 = nn.Conv1d(16, 24, 8)
        self.conv4 = nn.Conv1d(24, 32, 8)
        self.pool = nn.MaxPool1d(4, 2)

        x = torch.randn(binmax).view(-1,1,binmax)
        self.linear_len = None
        self.convs(x)

        self.fc1 = nn.Linear(self.linear_len, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, outF)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # This operation is to get the in-features for linear layer 1 (It will only run once)
        if self.linear_len is None:
            self.linear_len = len(torch.flatten(x))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linear_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class trainNet():
    def CrossVal(self, fold):
        self.train_accs = np.array([])
        self.test_accs = np.array([])
        self.conmats = []
        for i in range(fold):
            # Calling and loading the neural network
            print(f"Fold {i + 1}\n")
            self.net = CNN().to(device)
            testSize = int(len(X) * 1/fold)
            self.type_counter_train = np.zeros(7, dtype=int)
            self.type_counter_test = np.zeros(7, dtype=int)

            self.test_X = X[i * testSize : (i+1) * testSize]
            self.test_y = y[i * testSize : (i+1) * testSize]
            self.train_X = np.delete(X, np.s_[i * testSize : (i+1) * testSize], 0)
            self.train_y = np.delete(y, np.s_[i * testSize : (i+1) * testSize], 0)

            for label_train in self.train_y:
                self.type_counter_train[np.argmax(label_train)] += 1
            for label_test in self.test_y:
                self.type_counter_test[np.argmax(label_test)] += 1

            print(f"Size of train data:\n{len(self.train_y)}\nSize of test data\n{len(self.test_y)}")
            print(f"Train data distribution\n{self.type_counter_train}\nTest data distribution\n{self.type_counter_test}")
            self.train(EPOCHS=30)
            self.testAll()
        self.train_accs, self.test_accs = np.reshape(self.train_accs, (-1, 7)), np.reshape(self.test_accs, (-1, 7))

        figure = plt.figure(figsize=(18,6))
        figure.suptitle(f"{fold}-fold Cross Validation")

        ax1 = plt.subplot2grid((2, 4),(0, 0), rowspan=2)
        plt.bar(types, np.mean(self.train_accs, axis=0))
        plt.gca().set_ylim([0, 1])
        plt.title(f"Train Acc")

        ax2 = plt.subplot2grid((2, 4), (0, 1), rowspan=2)
        plt.bar(types, np.mean(self.test_accs, axis=0))
        plt.gca().set_ylim([0, 1])
        plt.title(f"Test Acc")

        self.conmats = np.mean(self.conmats, axis=0)
        self.conmats /= np.sum(self.conmats, axis=0)
        ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
        ax3.matshow(self.conmats, cmap="Blues")
        thresh = self.conmats.max() / 2.
        for j in range(7):
            for k in range(7):
                plt.gca().text(k, j, f"{self.conmats[j, k]:.2f}", va='center', ha='center',
                               color="white" if self.conmats[j, k] > thresh else "black",
                               weight="bold")
        plt.xlabel("True Class")
        ax3.xaxis.set_label_position('top')
        ax3.set_xticklabels([""]+types, rotation=90)
        ax3.xaxis.set_ticks_position('bottom')
        plt.ylabel("Predicted Class")
        ax3.set_yticklabels([""]+types)

        plt.savefig(f"figure/crossval_{rdf}_{name}_{binmax}_{EPOCHS}.png")
        plt.clf()

        return self.train_accs, self.test_accs

    def train(self, EPOCHS):
        batchSize = 100
        decayRate = 0.9
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        for epoch in range(EPOCHS): # 3 full passes over the data
            for i in range(0, len(self.train_X), batchSize):  # `data` is a batch of data
                batch_X = self.train_X[i:i+batchSize].view(-1, 1, binmax).to(device)
                batch_y = self.train_y[i:i+batchSize].to(device)
                self.net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = self.net(batch_X)  # pass in the reshaped batch
                loss = loss_function(output, batch_y)  # calc and grab the loss value
                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
            lr_scheduler.step()
            print(f"Epoch: {epoch + 1} / {EPOCHS}")
            print(loss)
        # torch.save(net.state_dict(), modelname)

    def testAll(self):
        confusion_matrix_test = np.zeros((7, 7), dtype=int)
        correct_counter_train = np.zeros(7, dtype=int)
        acc_train = np.zeros(7, dtype=float)
        correct_counter_test = np.zeros(7, dtype=int)
        acc_test = np.zeros(7, dtype=float)

        with torch.no_grad():
            for i in range(len(self.train_X)):
                label_train = torch.argmax(self.train_y[i].to(device))
                output_train = self.net(self.train_X[i].view(-1, 1, binmax).to(device))[0]
                prediction_train = torch.argmax(output_train)
                if label_train == prediction_train:
                    correct_counter_train[label_train] += 1

            for i in range(len(self.test_X)):
                label_test = torch.argmax(self.test_y[i].to(device))
                output_test = self.net(self.test_X[i].view(-1, 1, binmax).to(device))[0]
                prediction_test = torch.argmax(output_test)
                confusion_matrix_test[prediction_test, label_test] += 1
                if label_test == prediction_test:
                    correct_counter_test[label_test] += 1
            self.conmats.append(confusion_matrix_test)

            for i in range(len(correct_counter_train)):
                if self.type_counter_train[i] != 0:
                    acc_train[i] = correct_counter_train[i] / self.type_counter_train[i]
                if self.type_counter_test[i] != 0:
                    acc_test[i] = correct_counter_test[i] / self.type_counter_test[i]
            self.train_accs = np.append(self.train_accs, acc_train)
            self.test_accs = np.append(self.test_accs, acc_test)

if __name__ == "__main__":
    a = trainNet()
    train_accs, test_accs = a.CrossVal(fold=5)
    print(train_accs, np.mean(train_accs, axis=1))
    print(test_accs, np.mean(test_accs, axis=1))