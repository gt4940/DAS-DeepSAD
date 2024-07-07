import os
import torch
import random
import easydict
import numpy as np
from code.train import TrainerDeepSAD
from code.dataloader import make_augmented_dataloader, make_dataloader
from code.evaluation import load_sad_model, test_eval, roc_auc_score, test_score, plot_tsne_2D

# Fix seed
seed = 42
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # From error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({'num_epochs_pretrain': 350,
                          'lr_pretrain': 1e-3,
                          'weight_decay_pretrain': 1e-6,  # Weight regularization
                          'lr_milestones_pretrain': [300],  # Epoch(s) to change learning rate
                          'num_epochs': 300,
                          'lr': 5e-4,
                          'weight_decay': 1e-6,  # Weight regularization
                          'lr_milestones': [200],  # Epoch(s) to change learning rate
                          'num_filter': 16,
                          'latent_dim': 256,
                          'batch_size': 32,
                          'pretrain': True,
                          'image_height': 1000,  # For model summary
                          'image_width': 406,  # For model summary
                          'num_workers_dataloader': 6,
                          'output_path': './output/filter_16_latent_256/'
                          })


if __name__ == '__main__':
    os.mkdir(args.output_path)
    print(args.output_path)

    # Load Train/Test Loader
    dataloader_train = make_augmented_dataloader(data_dir='./data/das_image/', args=args, shuffle=True)
    dataloader_test = make_dataloader(data_dir='./data/deep_svdd_test/', args=args, shuffle=False)

    # Prepare to Learn Network
    deep_SAD = TrainerDeepSAD(args, dataloader_train, device)

    # Pretrain U-Net models for DeepSAD
    if args.pretrain:
        deep_SAD.pretrain()

    # Train Deep SAD model with pretrained U-Net weights
    deep_SAD.train()

    # Deep SAD model evaluation
    net_sad, c = load_sad_model(args, device)
    labels, scores, z_list, x_list = test_eval(net_sad, c, device, dataloader_test)
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)))
    result = test_score(scores, labels)
    plot_tsne_2D(x_list, z_list, labels, result)
