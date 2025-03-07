from unireplknet import *
import torch
import torch.nn as nn
import timm
from kan import KAN
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import EegDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
from data_aug import *
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

'''
model = model = timm.create_model('unireplknet_s')
model.load_state_dict(torch.load('models/UnirepCogNet.pth'))
model.head = KAN(width=[768, 3], grid=10, k=3, device='cuda')
# model.head.train(lamb=0.01)
x = torch.normal(-1, 1, size=(100, 3, 61, 500))
x = x.cuda()
model = model.cuda()
outs = model(x)
print(outs)
model.head.plot(beta=100)
plt.show()
'''


class CogRepLKNet(nn.Module):
    def __init__(self, num_classes=3, use_kan=False, pretrained=False, save_feature=False):
        super(CogRepLKNet, self).__init__()
        self.num_classes = num_classes
        self.UniRepLKNet = timm.create_model('unireplknet_s')
        self.Classifier = nn.Linear(128, 3)

        self.conv_embed = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        if pretrained:
            self.UniRepLKNet.load_state_dict(torch.load('models/UnirepCogNet.pth'))
        # self.UniRepLKNet.head.out_features = 128
        self.UniRepLKNet.head = nn.Linear(768, 128)

        if use_kan:
            self.UniRepLKNet.head = KAN(width=[768, 128], grid=5, k=3, device='cuda')
            self.Classifier = KAN(width=[128, 3], grid=5)

        self.save_feature = save_feature

    def forward(self, x):
        x = self.conv_embed(x)
        feature = self.UniRepLKNet(x)
        if self.save_feature:
            self.feature = feature
        outputs = self.Classifier(feature)
        return F.softmax(outputs, dim=1)

    def get_feature(self):
        if self.save_feature:
            return self.feature
        return None


class TrainAndEval:
    def __init__(self, data_path, log_path, epoch, lr, bs, save_model=False):
        self.data_path = data_path
        self.log_path = log_path
        self.epoch = epoch
        self.lr = lr
        self.bs = bs
        self.save_model = save_model

        # load data
        self.data = sio.loadmat(data_path)
        S1Data = self.data['S1Data']  # shape: (61, 500, 447)
        S1Data = np.transpose(S1Data, (2, 0, 1))
        S1Data = np.expand_dims(S1Data, axis=1)  # shape:(447, 1, 61, 500)
        S1Label = self.data['S1Label']  # shape: (447, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(S1Data, S1Label, test_size=0.2,
                                                                                random_state=42)
        self.y_train = np.squeeze(self.y_train)
        self.y_test = np.squeeze(self.y_test)
        self.train_dataset = EegDataset(self.X_train, self.y_train, 's')
        self.test_dataset = EegDataset(self.X_test, self.y_test, 's')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False)

        print("Info: Datapath: {}, Logpath: {}, Epoch: {}, lr: {}, bs: {}.".format(
            data_path, log_path, epoch, lr, bs
        ))

    def train_and_val(self, model, nSub):
        log_write = open(self.log_path + '/subject' + str(nSub) + '.txt', "w")
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        bestAcc = 0
        bestf1 = 0
        Y_true = 0
        Y_pred = 0
        best_kappa = 0
        model_dict = 0
        total_train_time = 0
        for epoch in range(self.epoch):
            # 遍历数据集
            model.train()
            total_loss = 0
            st = time.time()
            for i, (inputs, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{epoch + 1}")):
                # 将数据和标签转移到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
                # S&R 数据增强
                # if len(inputs) == batch_size:
                #    print(len(inputs))
                aug_data, aug_label = interaug(self.X_train, self.y_train, self.bs, 40)
                inputs = torch.cat((inputs, aug_data))
                labels = torch.cat((labels, aug_label))

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)
                # loss = loss + reg_loss
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # scheduler.step()
                total_loss = total_loss + loss.item()
            # 测试模型
            et = time.time()
            train_time = et - st
            total_train_time += train_time
            print("Train per epoch(357 samples): %.2f s" % train_time)
            model.eval()
            with torch.no_grad():
                # 初始化正确和总数
                correct = 0
                total = 0
                first = True
                # 遍历测试集
                for inputs, labels in self.test_loader:
                    # 将数据和标签转移到设备上
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = model(inputs)

                    # 预测类别
                    _, predicted = torch.max(outputs, 1)

                    # 更新正确和总数
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if first:
                        predy = predicted
                        truey = labels
                        first = False
                    else:
                        predy = torch.cat((predy, predicted))
                        truey = torch.cat((truey, labels))
                # 计算并打印准确率
                accuracy = 100 * correct / total
                f1 = 100 * f1_score(truey.cpu(), predy.cpu(), average='macro')
                kappa = 100 * cohen_kappa_score(truey.cpu(), predy.cpu())
            if accuracy > bestAcc:
                bestAcc = accuracy
                bestf1 = f1
                Y_true = truey
                Y_pred = predy
                best_kappa = kappa
                model_dict = model.state_dict()

            print(f'Epoch {epoch + 1}, Train Loss {total_loss / len(self.train_loader):.6f}, ',
                  f'Accuracy/macro-f1/kappa of the model on the test set: {accuracy:.2f}% / {f1:.2f}% / {kappa:.2f}%')
            log_write.write(str(epoch) + "   " + str(total_loss / len(self.train_loader)) + "   " + str(accuracy) +
                            "   " + str(f1) + "   " + str(kappa) + "   " + str(train_time) + "\n")
        # log_write.write('The best accuracy/f1-score is: ' + str(bestAcc) + "  " + str(bestf1) + "\n")
        print('The best accuracy/f1-score/kappa is: ' + str(bestAcc) + "  " + str(bestf1) + "  " + str(best_kappa))
        if self.save_model:
            print('Model saved !')
            torch.save(model_dict, self.log_path + "/model/" + str(nSub) + '.pth')
        path = self.log_path + '/cm/Subject' + str(nSub) + '.png'
        conf_matrix(Y_pred.cpu(), Y_true.cpu(), path, nSub, bestAcc, bestf1, best_kappa, total_train_time, log_write)
        return bestAcc, bestf1


def conf_matrix(pred_labels, true_labels, path, nsub, bestAcc, bestf1, best_kappa, total_train_time, log_write):
    cm = confusion_matrix(true_labels, pred_labels)

    # 提取每个类别的TP, TN, FP, FN
    sensitivities = []
    specificities = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    log_write.write(
        'The best acc/f1/kap/sen/spe/time is: ' + str(bestAcc) + "  " + str(bestf1) + "   " + str(best_kappa) +
        "  " + str(sensitivities[0]) + "  " + str(sensitivities[1]) + "  " + str(sensitivities[2]) +
        "  " + str(specificities[0]) + "  " + str(specificities[1]) + "  " + str(specificities[2]) +
        "  " + str(total_train_time) + "\n")
    log_write.close()

    # class_labels = ['Left_Hands', 'Right_Hands', 'Feet', 'Tongue']
    class_labels_r2 = ['Underload', 'Normal', 'Overload']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Blues
    plt.title('Subject ' + str(nsub))
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_r2))
    plt.xticks(tick_marks, class_labels_r2, rotation=45, fontsize=16)
    plt.yticks(tick_marks, class_labels_r2, fontsize=16)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if cm[i, j] < thresh else "white",
                 fontsize=20)

    plt.xlabel('Pred labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_multiclass_roc(truey, predy, classes, n_classes, path):
    truey_bin = label_binarize(truey, classes=range(n_classes))
    predy_bin = label_binarize(predy, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truey_bin[:, i], predy_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truey_bin.ravel(), predy_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-class')
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == '__main__':
    log_path = './results'
    data_path = '/home/sy/data/cognitive/Sub_S1_single/Sub_S1_'
    epoch = 10
    batch_size = 32
    learning_rate = 0.0003

    model = CogRepLKNet(pretrained=True)

    for i in range(1):
        data_path = data_path + str(i + 1)
        exp = TrainAndEval(data_path, log_path, epoch, learning_rate, batch_size, save_model=True)
        acc, f1 = exp.train_and_val(model, i + 1)
        print(acc, "   ", f1)
