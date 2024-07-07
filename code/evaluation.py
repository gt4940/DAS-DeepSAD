import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .network import DeepSAD
from matplotlib.colors import Normalize
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix


def load_sad_model(args, device):
    net = DeepSAD(args).to(device)
    net.eval()
    state_dict = torch.load(args.output_path + 'Deep_SAD_best_save.pth')
    net.load_state_dict(state_dict['net_dict'])
    c = torch.Tensor(state_dict['center']).to(device)
    return net, c


def test_eval(net, c, device, test_dataset, eps=1e-6):
    """Testing the Deep SVDD model"""
    scores = []
    labels = []
    x_list = []
    z_list = []
    test_loss = 0.0
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.float().to(device)
            z = net(x)
            dist = torch.sum((z - c) ** 2, dim=1)
            sad = torch.where(y == 0, dist, (dist + eps) ** y)
            loss_sad = torch.mean(sad)
            test_loss += loss_sad.item()

            x_list.append(x.detach().cpu())
            z_list.append(((z-c)**2).detach().cpu())
            scores.append(dist.detach().cpu())
            labels.append(y.cpu())

    print(f"Test Loss: {test_loss / len(test_dataset)}")
    x_list = torch.cat(x_list).numpy()
    z_list = torch.cat(z_list).numpy()
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = 0
        elif labels[i] == -1:
            labels[i] = 1
        else:
            print("error")

    return labels, scores, z_list, x_list


def get_tsne_2D(data, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=908)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


def get_tsne_3D(data, n_components=3):
    tsne = TSNE(n_components=n_components, random_state=908)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


def plot_tsne_2D(x_list, z_list, labels, scores, threshold):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 5))
    x_list = x_list.reshape((x_list.shape[0], 1000*406))

    tsne_data = get_tsne_2D(x_list)

    scatter = axs[0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels)
    handles, labels_before = scatter.legend_elements()
    axs[0].legend(handles=handles, labels=labels_before)
    axs[0].set_title('before')

    tsne_data = get_tsne_2D(z_list)

    cmap_ = plt.cm.get_cmap('seismic')
    noise = np.where(labels == 0)
    event = np.where(labels == 1)
    min_ = min(scores)
    scores[scores >= 2 * threshold - min_] = 2 * threshold - min_
    scores = scores - threshold

    min_, max_ = scores.min(), scores.max()
    scores = -1 + 2 * (scores - min_) / (max_ - min_)

    norm_ = Normalize(vmin=min(scores), vmax=max(scores))

    scatter = axs[1].scatter(tsne_data[noise, 0], tsne_data[noise, 1], marker='o', c=scores[noise], norm=norm_,
                             cmap=cmap_, label='Noise'
                             )
    scatter = axs[1].scatter(tsne_data[event, 0], tsne_data[event, 1], marker='*', c=scores[event], norm=norm_,
                             cmap=cmap_, label='Event'
                             )
    fig.colorbar(scatter, ax=axs[1])
    axs[1].legend()
    axs[1].set_title('after')


def test_score(scores, labels):
    fpr, tpr, thres = roc_curve(labels, scores)
    threshold = thres[np.argmax(tpr + (1-fpr))]
    print(threshold)
    print(scores)
    for i in range(len(scores)):
        if scores[i] <= threshold:
            scores[i] = 0
        elif scores[i] > threshold:
            scores[i] = 1
        else:
            print("error")
        print("count:", i, "scores:", scores[i], "labels", labels[i])

    confusion_mat = confusion_matrix(labels, scores)
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    print(confusion_mat)
    print(classification_report(labels, scores, output_dict=True)['weighted avg'])
    return scores
