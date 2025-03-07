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
from torch.utils.data import Subset
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import time

from data_aug import *
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from untils.early_stopping import EarlyStopping
from itertools import chain
from loss import SupConLoss
from fusion import AdaptiveGatedCAHead
import h5py

from utils.MTF import *


class CogRepLKNet(nn.Module):
    def __init__(self, in_feature=64,num_classes=2, use_kan=False, pretrained=False, save_feature=False, drop_rate=0.):
        super(CogRepLKNet, self).__init__()
        self.num_classes = num_classes
        self.UniRepLKNet = timm.create_model('unireplknet_a', drop_path_rate=drop_rate, in_chans=30)
        # self.Classifier = nn.Linear(128, 2)

        self.conv_embed = nn.Conv2d(in_channels=30, out_channels=3, kernel_size=1)

        if pretrained:
            self.UniRepLKNet.load_state_dict(torch.load('models/unireplknet_a.pth'))
        # self.UniRepLKNet.head.out_features = 128
        self.UniRepLKNet.head = nn.Linear(in_feature, 64) # (192, 128)
        self.UniRepLKNet.norm = nn.LayerNorm(in_feature) # 192

        if use_kan:
            self.UniRepLKNet.head = KAN(width=[192, 128], grid=5, k=3, device='cuda')
        #     self.Classifier = KAN(width=[128, 2], grid=5, k=3, device='cuda')

        self.save_feature = save_feature

    def forward(self, x):
        # x = self.conv_embed(x)
        feature = self.UniRepLKNet(x)
        if self.save_feature:
            self.feature = feature
        # outputs = self.Classifier(feature)
        # return F.softmax(outputs, dim=1)
        return feature

    def get_feature(self):
        if self.save_feature:
            return self.feature
        return None

class CogRepLKNet_mri(nn.Module):
    def __init__(self, in_feature=64,num_classes=2, use_kan=False, pretrained=False, save_feature=False, drop_rate=0.):
        super(CogRepLKNet_mri, self).__init__()
        self.num_classes = num_classes
        self.UniRepLKNet = timm.create_model('unireplknet_a', drop_path_rate=drop_rate, in_chans=1)
        # self.Classifier = nn.Linear(128, 2)

        self.conv_embed = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        if pretrained:
            self.UniRepLKNet.load_state_dict(torch.load('models/unireplknet_a.pth'))
        # self.UniRepLKNet.head.out_features = 128
        self.UniRepLKNet.head = nn.Linear(in_feature, 64) # (192, 128)
        self.UniRepLKNet.norm = nn.LayerNorm(in_feature) # 192

        if use_kan:
            self.UniRepLKNet.head = KAN(width=[192, 128], grid=5, k=3, device='cuda')
        #     self.Classifier = KAN(width=[128, 2], grid=5, k=3, device='cuda')

        self.save_feature = save_feature

    def forward(self, x):
        # x = self.conv_embed(x)
        feature = self.UniRepLKNet(x)
        if self.save_feature:
            self.feature = feature
        # outputs = self.Classifier(feature)
        # return F.softmax(outputs, dim=1)
        return feature

    def get_feature(self):
        if self.save_feature:
            return self.feature
        return None

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, use_KAN=False):
        super(ClassificationHead, self).__init__()
        if use_KAN:
            self.fc1 = KAN(width=[in_features, 64], grid=5, k=3, device='cuda')
            self.fc2 = KAN(width=[64, num_classes], grid=5, k=3, device='cuda')
        else:
            self.fc1 = nn.Linear(in_features, 64)
            self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class TrainAndEval:
    def __init__(self, data_path, data_path_r, log_path, epoch, lr, bs, save_model=False):
        self.data_path = data_path
        self.data_path_r = data_path_r
        self.log_path = log_path
        self.epoch = epoch
        self.lr = lr
        self.bs = bs
        self.save_model = save_model

        # load EEG data
        with h5py.File(data_path + '.mat', 'r') as f:
            self.data = f['data1'][:]
            self.label = f['label1'][:]
        S1Data = np.transpose(self.data, (0, 2, 1))  # shape: (480, 30, 500)
        # S1Data = np.transpose(S1Data, (2, 0, 1))
        MTF_images = np.zeros((S1Data.shape[0], 30, 256, 256))

        for i in tqdm(range(S1Data.shape[0]), desc="Generating MTF Images"):
            sample = S1Data[i]  # 提取第i个样本, shape: (61, 500)
            MTF_image = getMTF(sample)  # 生成MTF图像, shape: (128, 128)
            MTF_images[i, :, :, :] = MTF_image  # 填充到MTF_images中
        
        S1Label = np.where(self.label == 11, 0,
            np.where(self.label == 12, 1,
            np.where(self.label == 13, 2,
            np.where(self.label == 14, 3, -1))))  # -1 表示未匹配的默认值
        S1Label = np.transpose(S1Label, (1, 0))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(MTF_images, S1Label, test_size=0.2,
                                                                                random_state=42)
        # load fMRI data
        with h5py.File(data_path_r + '.mat', 'r') as f:
            self.data_r = f['data'][:]
        # S1Data_r = self.data_r['data']  # shape: (116, 4, 240)
        # S1Data_r = np.transpose(S1Data_r, (2, 1, 0))
        S1Data_r = self.data_r
        S1Data_r = np.expand_dims(S1Data_r, axis=1)
        self.X_train_r, self.X_test_r, self.y_train_r, self.y_test_r = train_test_split(S1Data_r, S1Label, test_size=0.2,
                                                                                        random_state=42)
        # shuffle data
        self.shuffle_num = np.random.permutation(len(self.X_train))  # 生成一个随机索引 288
        self.X_train = self.X_train[self.shuffle_num]
        self.y_train = self.y_train[self.shuffle_num]
        self.X_train_r = self.X_train_r[self.shuffle_num]
        self.y_train_r = self.y_train_r[self.shuffle_num]
        
        '''
        target_mean = np.mean(S1Data)
        target_std = np.std(S1Data)
        self.X_train = (self.X_train - target_mean) / target_std
        self.X_test = (self.X_test - target_mean) / target_std
        target_mean = np.mean(S1Data_r)
        target_std = np.std(S1Data_r)
        self.X_train_r = (self.X_train_r - target_mean) / target_std
        self.X_test_r = (self.X_test_r - target_mean) / target_std
        '''
        self.y_train = np.squeeze(self.y_train)
        self.y_test = np.squeeze(self.y_test)
        self.train_dataset = EegDataset(self.X_train, self.y_train, 's')
        self.test_dataset = EegDataset(self.X_test, self.y_test, 's')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False)

        self.train_dataset_r = EegDataset(self.X_train_r, self.y_train, 's')
        self.test_dataset_r = EegDataset(self.X_test_r, self.y_test, 's')
        self.train_loader_r = torch.utils.data.DataLoader(self.train_dataset_r, batch_size=self.bs, shuffle=False)
        self.test_loader_r = torch.utils.data.DataLoader(self.test_dataset_r, batch_size=self.bs, shuffle=False)

        print("Info: Datapath: {}, Logpath: {}, Epoch: {}, lr: {}, bs: {}.".format(
            data_path, log_path, epoch, lr, bs
        ))
        print(self.X_train.shape)
        print(self.X_train_r.shape)

    def train_and_val(self, model, model_r, class_head, nSub):
        log_write = open(self.log_path + '/subject' + str(nSub) + '.txt', "w")
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = model.to(device)
        model_r = model_r.to(device)
        class_head = class_head.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        criterion_sc = SupConLoss(use_device=device) # The loss to regular multi-modal features

        optimizer = torch.optim.Adam(chain(model.parameters(), model_r.parameters(), class_head.parameters()), lr=self.lr)

        early_stopping = EarlyStopping(patience=10, verbose=True)

        bestAcc = 0
        bestf1 = 0
        Y_true = 0
        Y_pred = 0
        Y_prob = 0
        best_kappa = 0
        model_dict = 0
        total_train_time = 0
        for epoch in range(self.epoch):
            # 遍历数据集
            indices = torch.randperm(len(self.train_dataset)).tolist()
            self.train_loader = torch.utils.data.DataLoader(Subset(self.train_dataset, indices), batch_size=self.bs, shuffle=False)
            self.train_loader_r = torch.utils.data.DataLoader(Subset(self.train_dataset_r, indices), batch_size=self.bs, shuffle=False)
            model.train()
            model_r.train()
            class_head.train()
            total_loss = 0
            st = time.time()
            tacc = 0
            for i, ((inputs, labels), (inputs_r, labels_r)) in enumerate(tqdm(zip(self.train_loader, self.train_loader_r), desc=f"Epoch {self.epoch}/{epoch + 1}")):
                # 将数据和标签转移到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs_r = inputs_r.to(device)
                # S&R 数据增强
                # if len(inputs) == batch_size:
                #    print(len(inputs))
                # aug_data, aug_label = interaug(self.X_train, self.y_train, self.bs, 40, 4, device)
                # inputs = torch.cat((inputs, aug_data))
                # labels = torch.cat((labels, aug_label))

                # aug_data_r, _ = finteraug(self.X_train_r, self.y_train, self.bs, 40, 4, device)
                # inputs_r = torch.cat((inputs_r, aug_data_r))

                # 前向传播
                optimizer.zero_grad()
                features = model(inputs)
                features_r = model_r(inputs_r)
                multi_features = torch.cat((features, features_r), 1)
                outputs = class_head(features, features_r)

                # 计算损失
                loss_ce = criterion(outputs, labels)
                tacc = tacc + (torch.max(outputs, 1)[1] == labels).sum().item()
                # 计算SC损失
                multi_modal_features = torch.cat((features.unsqueeze(-1), features_r.unsqueeze(-1)))
                multi_modal_labels = torch.cat((labels, labels), dim=0)
                loss_sc = criterion_sc(multi_modal_features, multi_modal_labels)
                
                loss = loss_ce + 0.2 * loss_sc # namuta = 0.5
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # scheduler.step()
                total_loss = total_loss + loss.item()
            # 测试模型
            et = time.time()
            train_time = et - st
            total_train_time += train_time
            #print("Train per epoch(357 samples): %.2f s" % train_time)
            model.eval()
            model_r.eval()
            class_head.eval()
            with torch.no_grad():
                # 初始化正确和总数
                correct = 0
                total = 0
                first = True
                # 遍历测试集
                for (inputs, labels), (inputs_r, labels_r) in zip(self.test_loader, self.test_loader_r):
                    # 将数据和标签转移到设备上
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs_r = inputs_r.to(device)

                    # 前向传播
                    features = model(inputs)
                    features_r = model_r(inputs_r)
                    multi_features = torch.cat((features, features_r), 1)
                    outputs = class_head(features, features_r)

                    # 预测类别
                    _, predicted = torch.max(outputs, 1)

                    # 更新正确和总数
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if first:
                        predy = predicted
                        truey = labels
                        proby = outputs
                        first = False
                    else:
                        predy = torch.cat((predy, predicted))
                        truey = torch.cat((truey, labels))
                        proby = torch.cat((proby, outputs))
                # 计算并打印准确率
                accuracy = 100 * correct / total
                f1 = 100 * f1_score(truey.cpu(), predy.cpu(), average='macro')
                kappa = 100 * cohen_kappa_score(truey.cpu(), predy.cpu())
            if accuracy > bestAcc:
                bestAcc = accuracy
                bestf1 = f1
                Y_true = truey
                Y_pred = predy
                Y_prob = proby
                best_kappa = kappa
                model_dict = model.state_dict()
                model_dict_r = model_r.state_dict()
                class_head_dict = class_head.state_dict()

            # early_stopping(total_loss / len(self.train_loader))
            print(f'Epoch {epoch + 1}, Train Loss {total_loss / len(self.train_loader):.6f}, {tacc / 768: .2f}, ',
                  f'Accuracy/macro-f1/kappa of the model on the test set: {accuracy:.2f}% / {f1:.2f}% / {kappa:.2f}%')
            log_write.write(str(epoch) + "   " + str(total_loss / len(self.train_loader)) + "   " + str(accuracy) +
                            "   " + str(f1) + "   " + str(kappa) + "   " + str(train_time) + "\n")
            # if early_stopping.early_stop:
            #     print("Early stopping!")
            #     break
        # log_write.write('The best accuracy/f1-score is: ' + str(bestAcc) + "  " + str(bestf1) + "\n")
        print('The best accuracy/f1-score/kappa is: ' + str(bestAcc) + "  " + str(bestf1) + "  " + str(best_kappa))
        if self.save_model:
            print('Model saved!')
            torch.save(model_dict, self.log_path + "/model/model" + str(nSub) + '.pth')
            torch.save(model_dict_r, self.log_path + "/model/modelr" + str(nSub) + '.pth')
            torch.save(class_head_dict, self.log_path + "/model/class_head" + str(nSub) + '.pth')
        path = self.log_path + '/cm/Subject' + str(nSub) + '.png'
        path_roc = self.log_path + '/roc/Subject' + str(nSub) + '.png'
        conf_matrix(Y_pred.cpu(), Y_true.cpu(), path, nSub, bestAcc, bestf1, best_kappa, total_train_time, log_write)
        # plot_multiclass_roc_class2(Y_true.cpu(), Y_prob.cpu(), path_roc)
        # save Y_prob and Y_true
        # print("Save Probability and True label? (Yes/No): ", end="")
        # s = input()
        # if s != 'No':
        torch.save(Y_prob, self.log_path + '/probability/prob' + str(nSub) + '.pth')
        torch.save(Y_true, self.log_path + '/probability/true' + str(nSub) + '.pth')
        print("Probability and True label saved!")
        return bestAcc, bestf1


def conf_matrix(pred_labels, true_labels, path, nsub, bestAcc, bestf1, best_kappa, total_train_time, log_write):
    cm = confusion_matrix(true_labels, pred_labels)

    # 提取每个类别的TP, TN, FP, FN
    sensitivities = []
    specificities = []
    precisions = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        sensitivities.append(sensitivity * 100)
        specificities.append(specificity * 100)
        precisions.append(precision)

    log_write.write(
        'The best acc/f1/kap/precision/sen/spe/time is: ' + str(bestAcc) + "  " + str(bestf1) + "   " + str(best_kappa) + "  " + str(np.mean(precision) * 100) +
        "  " + str(sensitivities[0]) + "  " + str(sensitivities[1]) +  "  " + str(sensitivities[2]) + "  " + str(sensitivities[3]) +
        "  " + str(specificities[0]) + "  " + str(specificities[1]) +  "  " + str(specificities[2]) + "  " + str(sensitivities[3]) +
        "  " + str(total_train_time) + "\n")
    '''
    log_write.write(
        'The best acc/f1/kap/recall/precision/time is: ' + str(bestAcc) + "  " + str(bestf1) + "   " + str(best_kappa) +
        "  " + str(sensitivities[0] * 100) + "  " + str(precisions[0] * 100) +
        "  " + str(total_train_time) + "\n")
    log_write.close()
    '''
    # class_labels = ['Left_Hands', 'Right_Hands', 'Feet', 'Tongue']
    class_labels_r2 = ['Underload', 'Normal', 'Overload']
    class_labels_r3 = ['Low Cognition', 'High Cognitive']
    class_labels_r4 = ['Level-1', 'Level-2', 'Level-3', 'Level-4']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Blues
    plt.title('Subject ' + str(nsub), fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_r4))
    plt.xticks(tick_marks, class_labels_r4, rotation=45, fontsize=15)
    plt.yticks(tick_marks, class_labels_r4, fontsize=15)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if cm[i, j] < thresh else "white",
                 fontsize=20)

    plt.xlabel('Pred labels', fontsize=15)
    plt.ylabel('True labels', fontsize=15)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_multiclass_roc(truey, predy, n_classes, path):
    classes = ['Underload', 'Normal', 'Overload']
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
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.2f})'
                                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic to Multi-class', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_multiclass_roc_class2(truey, predy, path):
    # 计算类别 1 的预测概率
    pred_prob_class1 = predy[:, 1]

    # 计算 ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(truey, pred_prob_class1)
    roc_auc = auc(fpr, tpr)

    # 初始化绘图
    plt.figure()

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')

    # 绘制对角线（随机分类器的表现）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图形参数
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curve for Binary Classification', fontsize=16)
    plt.legend(loc="lower right")

    # 保存图形
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == '__main__':
    log_path = './results'
    data_path = '/home/sy/data/multi_model/EEG/Sub_S1_single_4/Sub_S1_'
    data_path_r = '/home/sy/data/multi_model/fMRI/Sub_S1_single_4/Sub_S1_'
    epoch = 200
    batch_size = 32
    learning_rate = 0.001

    model = CogRepLKNet(in_feature=80, pretrained=False, drop_rate=0.2)
    model_r = CogRepLKNet_mri(in_feature=80, pretrained=False, drop_rate=0.2)
    class_head = ClassificationHead(64 * 2, 4) # (128 * 2)
    AGCA_head = AdaptiveGatedCAHead(num_classes=4)
    # torch加载上面三个模型的参数
    # model.load_state_dict(torch.load('results/model/model25.pth'))
    # model_r.load_state_dict(torch.load('results/model/modelr25.pth'))
    # AGCA_head.load_state_dict(torch.load('results/model/class_head24.pth'))
    # print(model)
    # 6-10, 16, 18, 20, 21, 24, 25, 27, 29-34
    # r = list(range(5, 11)) + [15, 17, 19, 20, 23, 24, 26] + list(range(28, 34)) # list(range(5, 10)) + [15, 17, 19, 20, 23, 24, 26] + list(range(28, 34))
    r = list(range(11, 23)) + list(range(24, 35))
    for i in r:
        data_path_sub = data_path + str(i + 1)
        data_path_sub_r = data_path_r + str(i + 1)
        exp = TrainAndEval(data_path_sub, data_path_sub_r, log_path, epoch, learning_rate, batch_size, save_model=True)
        acc, f1 = exp.train_and_val(model, model_r, AGCA_head, i + 1)
        print(acc, "   ", f1)
