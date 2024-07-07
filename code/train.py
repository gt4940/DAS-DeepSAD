import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from .network import DeepSAD, UNet
from torch.utils.tensorboard import SummaryWriter


class TrainerDeepSAD:
    def __init__(self, args, dataloader_train, device):
        self.args = args
        self.train_loader = dataloader_train
        self.device = device
        self.writer = SummaryWriter(log_dir=self.args.output_path)

    def pretrain(self):
        pretrain_net = UNet(self.args).to(self.device)
        pretrain_net.apply(weights_init_normal)
        optimizer = torch.optim.Adam(pretrain_net.parameters(),
                                     lr=self.args.lr_pretrain,
                                     weight_decay=self.args.weight_decay_pretrain
                                     )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones_pretrain,
                                                         gamma=0.1
                                                         )
        print(summary(pretrain_net, (self.args.batch_size, 1, self.args.image_height, self.args.image_width),
                      row_settings=["var_names"]
                      )
              )
        pretrain_net.train()

        losses = []
        for epoch in range(self.args.num_epochs_pretrain):
            start = time.time()
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = pretrain_net(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            loss_temp = total_loss/len(self.train_loader)
            self.writer.add_scalar('Loss/U-Net', loss_temp, epoch+1)
            losses.append(loss_temp)
            if loss_temp <= min(losses):
                c = self.set_c(pretrain_net, self.train_loader)
                torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': pretrain_net.state_dict()},
                           self.args.output_path+'pretrain_best_save.pth'
                           )
                print(f'save best model at {epoch+1} epoch')
            print(f'Pretraining U-Net... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        plot_image(self.args, losses, 'U-Net')
        self.save_weights_for_DeepSAD(pretrain_net)

    def save_weights_for_DeepSAD(self, model):
        state_dict = torch.load(self.args.output_path+'pretrain_best_save.pth')
        model.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
        net = DeepSAD(self.args).to(self.device)
        net.load_state_dict(state_dict['net_dict'], strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()},
                   self.args.output_path+'pretrained_SAD.pth'
                   )

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        model.train()
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self, eps=1e-6):
        net = DeepSAD(self.args).to(self.device)
        if self.args.pretrain is True:
            state_dict = torch.load(self.args.output_path+'pretrained_SAD.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        print(summary(net, (self.args.batch_size, 1, self.args.image_height, self.args.image_width)))
        net.train()
        losses = []
        for epoch in range(self.args.num_epochs):
            start = time.time()
            total_loss = 0
            for x, y in self.train_loader:
                x = x.float().to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                z = net(x)
                dist = torch.sum((z - c) ** 2, dim=1)
                sad = torch.where(y == 0, dist, (dist + eps) ** y)
                loss = torch.mean(sad)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            loss_temp = total_loss/len(self.train_loader)
            self.writer.add_scalar('Loss/Deep SAD', loss_temp, epoch+1)
            losses.append(loss_temp)

            if loss_temp <= min(losses):
                net.eval()
                torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()},
                           self.args.output_path+'Deep_SAD_best_save.pth'
                           )
                net.train()
                print(f'save best Deep_SAD model at {epoch+1} epoch')
            print(f'Training Deep SAD... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        self.writer.flush()
        self.writer.close()
        plot_image(self.args, losses, 'Deep SAD')


def plot_image(args, losses, type):
    losses = np.array(losses)
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.plot(losses)
    plt.savefig(args.output_path + type + '_Loss.png', format='png')
    np.save(args.output_path + type + '_Loss.npy', losses)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
