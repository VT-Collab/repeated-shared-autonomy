import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from train_classifier import Net
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Classifer
class Model(object):

    def __init__(self, modelname):
        self.model = Net()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        self.enable_dropout()

    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def classify(self, x):
        return self.model.classifier(x)

    def forward(self, x):
        c = x[0]
        s = x[1]
        c_output = self.classify(c)
        c_true = x[4]
        loss = self.loss(c_output, c_true)
        return loss

    def loss(self, output, target):
        return self.loss_func(output, target)


# train cAE
def main():

    name = "models/classifier_dist"
    model = Model(name)

    X_train = pickle.load(open("data/X_train_data.pkl", "rb"))
    y_train = pickle.load(open("data/y_train_data.pkl", "rb"))
    X_test = pickle.load(open("data/X_test_data.pkl", "rb"))
    y_test = pickle.load(open("data/y_test_data.pkl", "rb"))

    correct = 0
    tn = 0
    fp = 0
    true_negatives = []
    false_positives = []

    with torch.no_grad():
        for idx, data in enumerate(X_test):
            c = torch.FloatTensor(data[0]).to(device)
            out = model.classify(c)
            _, label = torch.max(out.data, 0)
            if label.item() == y_test[idx]:
                correct += 1
            elif label.item() == 0:
                tn += 1
                true_negatives.append(data)
            elif label.item() == 1:
                fp += 1
                false_positives.append(data)

    y_test = np.asarray(y_test)
    true_count =  np.count_nonzero(y_test == 0)
    false_count =  np.count_nonzero(y_test == 1)
    print("total true: ", true_count)
    print("total false:", false_count)
    print("Accuracy: {} => {}".format(len(y_test), correct/float(len(y_test))) )
    print("True negatives: {} => {}".format(tn , tn/float(len(y_test))) )
    print("False positives: {} => {}".format(fp, fp/float(len(y_test))))


    plot_tn = random.sample(true_negatives, 9)
    plot_tn = [element[0] for element in plot_tn]
    # print(plot_tn[0][:2])
    fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharey=True)
    i = 0
    for ax in axs.reshape(-1):
        position_blue = plot_tn[i][:2]
        position_green = plot_tn[i][2:4]
        position_start = plot_tn[i][6:8]
        position_current = plot_tn[i][8:]
        ax.plot(position_blue[0], position_blue[1], 'bo', markersize=14)
        ax.plot(position_green[0], position_green[1], 'go', markersize=14)
        ax.plot(position_start[0], position_start[1], 'ko', markersize=14)
        ax.plot(position_current[0], position_current[1], 'mo', markersize=14)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, 1.0) 
        i+= 1
    plt.suptitle("True Negatives")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()