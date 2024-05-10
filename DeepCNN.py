from memory_profiler import profile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
nsys_executable = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2023.2.3/target-windows-x64/nsys.exe"
LEARNING_RATE = 0.0005
EPOCHS = 1
BATCH_SIZE = 16
data_path = 'E:/data/emotion2/Preprocessed_EEG/Sub_S1_single/Sub_S1_1.mat'
eeg_data_name = 'S1Data'
labels_name = 'S1Label'
num_classes = 3


# define data_set loader
class EegDataset(Dataset):
    def __init__(self, data, labels, source_or_target, transform=None):
        self.transform = transform
        if source_or_target == 's':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()
        elif source_or_target == 't':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.eeg_data[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x, y


class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.num_classes = num_classes

        # Conv Pool Block-1
        self.conv11 = nn.Conv2d(1, 25, (1, 10), padding=0)
        self.conv12 = nn.Conv2d(25, 25, (62, 1), padding=0)  # [bs,25,61,491]->[bs,25,1,491] and next->[bs,1,25,491]
        self.bn1 = nn.BatchNorm2d(25, False)
        self.pooling1 = nn.AvgPool2d((1, 3), stride=(1, 3))

        # Conv Pool Block-2
        self.conv2 = nn.Conv2d(25, 50, (1, 10), padding=0)  # [bs,1,25,163]->[bs,50,1,154] and next->[bs,1,50,154]
        self.bn2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.AvgPool2d((1, 3), stride=(1, 3))

        # Conv Pool Block-3
        self.conv3 = nn.Conv2d(50, 100, (1, 10), padding=0)  # [bs,1,50,51]->[bs,100,1,42] and next->[bs,1,100,42]
        self.bn3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.AvgPool2d((1, 3), stride=(1, 2))

        # Conv Pool Block-4
        self.conv4 = nn.Conv2d(100, 200, (1, 10), padding=0)  # [bs,1,100,14]->[bs,200,1,5] and next->[bs,1,200,5]
        self.bn4 = nn.BatchNorm2d(200)
        self.pooling4 = nn.AvgPool2d((1, 2), stride=(1, 2))

        # Linear classification
        self.fc1 = nn.Linear(600, self.num_classes)

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.25)
    def forward(self, x):
        # Layer 1
        x = self.pooling1(self.relu(self.bn1(self.conv12(self.conv11(x)))))
        x = self.dp(x)

        # Layer 2
        x = self.pooling2(self.relu(self.bn2(self.conv2(x))))
        x = self.dp(x)

        # Layer 3
        x = self.pooling3(self.relu(self.bn3(self.conv3(x))))
        x = self.dp(x)

        # Layer 4
        x = self.pooling4(self.relu(self.bn4(self.conv4(x))))
        x = self.dp(x)

        # 全连接层
        # x.shape=[32,1,200,1]
        x = x.view(x.size(0), -1)
        x = nn.Softmax(dim=1)(self.fc1(x))
        return x


def conf_matrix(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    # print(cm)
    class_labels = ['Underload', 'Normal', 'Overload', 'HV']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Blues
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()


# Load data
# Here, you should load your EEG data and labels into train_loader and test_loader
def load_data():
    data = scipy.io.loadmat(data_path)
    eeg_data = data[eeg_data_name]
    labels = data[labels_name] + 1
    # eeg_data = np.transpose(eeg_data,(2,0,1))

    train_set, test_set, train_label, test_label = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    data_set_train = EegDataset(train_set, train_label, 's')
    data_set_test = EegDataset(test_set, test_label, 's')
    train_loader = DataLoader(data_set_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(data_set_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_label, test_label


def train_and_val(train_loader, test_loader, train_label, test_label):
    # Define the model
    model = DeepCNN(num_classes=num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    bestacc = 0
    bestf1 = 0
    bestpred = 0
    bestlabel = 0
    model = model.cuda()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.unsqueeze(0).permute(1, 0, 2, 3)
            inputs, labels = inputs.cuda(), labels.cuda()
            # optimizer = lr_scheduler(optimizer, epoch)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d], Loss: %.4f  ' % (epoch + 1, running_loss / len(train_loader)), end='')
        # Test the model
        correct = 0
        total = 0
        strat_test = True
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(0).permute(1, 0, 2, 3)
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                if strat_test:
                    all_output = predicted
                    strat_test = False
                else:
                    all_output = torch.cat((all_output, predicted), dim=0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(predicted.shape)
            print('Accuracy of the network on the test set: %.2f %% , F1: %.2f %%' % (
                100 * correct / total, 100.0 * f1_score(test_label, all_output.cpu(), average='weighted')))
        if bestacc < correct / total:
            bestacc = correct / total
            bestf1 = f1_score(test_label, all_output.cpu(), average='weighted')
            bestpred = all_output
            bestlabel = test_label
    print('Best acc: %f %%, f1: %f %%' % (bestacc * 100, bestf1 * 100))
    conf_matrix(bestpred.cpu(), bestlabel)


# Main function
if __name__ == "__main__":
    train_loader, test_loader, train_label, test_label = load_data()
    train_and_val(train_loader, test_loader, train_label, test_label)
