import numpy as np
import torch


def interaug(timg, label, batch_size, num, num_classes=2, device=torch.device('cuda:0')):
    aug_data = []
    aug_label = []
    for cls4aug in range(num_classes):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        tmp_aug_data = np.zeros((int(batch_size / num_classes), 1, 30, 500))
        for ri in range(int(batch_size / num_classes)):
            for rj in range(5):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                tmp_aug_data[ri, :, :, rj * 100:(rj + 1) * 100] = tmp_data[rand_idx[rj], :, :,
                                                                  rj * 100:(rj + 1) * 100]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / num_classes)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).to(device)
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).to(device)
    aug_label = aug_label.long()
    # return aug_data, aug_label
    return aug_data[:num], aug_label[:num]

def finteraug(timg, label, batch_size, num, num_classes=2, device=torch.device('cuda:0')):
    aug_data = []
    aug_label = []
    for cls4aug in range(num_classes):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        tmp_aug_data = np.zeros((int(batch_size / num_classes), 1, 4, 116))
        for ri in range(int(batch_size / num_classes)):
            for rj in range(4):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 4)
                tmp_aug_data[ri, :, :, rj * 29:(rj + 1) * 29] = tmp_data[rand_idx[rj], :, :,
                                                                    rj * 29:(rj + 1) * 29]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / num_classes)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).to(device)
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).to(device)
    aug_label = aug_label.long()
    # return aug_data, aug_label
    return aug_data[:num], aug_label[:num]