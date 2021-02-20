import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm, trange
import numpy as np

device = torch.device('cuda')

class Rotate6D2Matrix(nn.Module):
    def __init__(self):
        super(Rotate6D2Matrix, self).__init__()
    def forward(self, a):
        a1, a2 = a[:, :, 0], a[:, :, 1]
        b1 = a1
        b1 = b1 / torch.sum(b1**2, dim=-1, keepdim=True)**.5
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = b2 / torch.sum(b2**2, dim=-1, keepdim=True)**.5
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=2)

class HandConv(nn.Module):
    def __init__(self, out_channels):
        super(HandConv, self).__init__()
        self.l = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        self.r = [i + 1 for i in self.l]
        self.k = nn.Parameter(torch.zeros(6, out_channels))
        nn.init.kaiming_uniform_(self.k)
    def forward(self, h):
        # print(torch.cat((h[:, self.l, :], h[:, self.r, :]), dim=-1).shape)
        return torch.cat((h[:, self.l, :], h[:, self.r, :]), dim=-1) @ self.k[None]


class ImitateNet(nn.Module):
    def __init__(self, in_dim, conv_channels, conv_dim, hidden_dim, dropout_rate):
        super(ImitateNet, self).__init__()
        self.conv = HandConv(conv_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(conv_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 6, bias=True)
        self.fc3 = nn.Linear(hidden_dim, 3, bias=True)
        self.rotate = Rotate6D2Matrix()
        for module in self.modules():
            if module is nn.Linear:
                nn.init.kaiming_uniform_(model.weight)
                if model.bias is not None:
                    nn.init.normal_(model.bias)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        rotate = self.rotate(self.fc2(x).reshape(-1, 3, 2))
        trans = self.fc3(x)
        return rotate, trans

class GraspLoss(nn.Module):
    def __init__(self):
        super(GraspLoss, self).__init__()
    def forward(self, p_rotate, p_trans, t_rotate, t_trans):
        return (((p_rotate - t_rotate)**2).sum() + ((p_trans - t_trans)**2).sum() / 3) / p_rotate.size(0)

def eval(model, norm, hand):
    single = False
    if len(hand.shape) == 2:
        single = True
        hand = hand[None, :]
    model = torch.load(model).cpu()
    model.eval()
    norm = np.load(norm)
    mean, std = norm['trans_mean'], norm['trans_std']
    with torch.no_grad():
        rotate, trans = model(torch.Tensor(hand))
        trans = trans * std + mean
        if single:
            return rotate.numpy()[0], trans.numpy()[0]
        else:
            return rotate.numpy(), trans.numpy()

def eval_loss(model, norm, hand, t_rotate, t_trans):
    if len(hand.shape) == 2:
        hand = hand[None, :]
    model = torch.load(model).cpu()
    model.eval()
    norm = np.load(norm)
    mean, std = norm['trans_mean'], norm['trans_std']
    t_trans = (t_trans - mean) / std
    with torch.no_grad():
        rotate, trans = model(torch.Tensor(hand))
        return GraspLoss()(rotate, trans, t_rotate, t_trans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int, default=63)
    parser.add_argument('--conv_channels', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=16 * 64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train', type=str)
    parser.add_argument('--val', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--save', type=str, default='saved_models/model.pkl')
    parser.add_argument('--save_norm', type=str, default='saved_models/norm.npz')
    parser.add_argument('--log_dir', type=str, default='log')
    args = parser.parse_args()

    model = ImitateNet(args.in_dim, args.conv_channels, args.conv_dim, args.hidden_dim, args.dropout).to(device=device)
    writer = SummaryWriter(args.log_dir)

    train_data = np.load(args.train)
    train_hand = torch.from_numpy(train_data['hand']).type(torch.float32)
    train_grasp_rotate = torch.from_numpy(train_data['grasp_rotate']).type(torch.float32)
    train_grasp_trans = torch.from_numpy(train_data['grasp_trans']).type(torch.float32)
    trans_mean = train_grasp_trans.mean(dim=0, keepdim=True)
    trans_std = train_grasp_trans.std(dim=0, keepdim=True)
    train_grasp_trans = (train_grasp_trans - trans_mean) / trans_std
    train_len = train_hand.size(0)

    val_data = np.load(args.val)
    val_hand = torch.from_numpy(val_data['hand'])
    val_grasp_rotate = torch.from_numpy(val_data['grasp_rotate'])
    val_grasp_trans = torch.from_numpy(val_data['grasp_trans'])
    val_grasp_trans = (val_grasp_trans - trans_mean) / trans_std
    val_len = val_hand.size(0)

    np.savez(args.save_norm, trans_mean=trans_mean, trans_std=trans_std)

    torch_train_data = torch.utils.data.TensorDataset(train_hand, train_grasp_rotate, train_grasp_trans)
    torch_train_iter = torch.utils.data.DataLoader(torch_train_data, batch_size=args.batch_size, shuffle=True)
    torch_val_data = torch.utils.data.TensorDataset(val_hand, val_grasp_rotate, val_grasp_trans)
    torch_val_iter = torch.utils.data.DataLoader(torch_val_data, batch_size=args.batch_size)
    torch_loss = GraspLoss()
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch_scheduler = torch.optim.lr_scheduler.StepLR(torch_optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    epoch_tqdm = trange(args.epochs)
    batch_loss, train_loss, val_loss = 0, 0, 0
    for e in epoch_tqdm:
        model.train()
        loss_sum = 0
        for hand, grasp_rotate, grasp_trans in torch_train_iter:
            pred_rotate, pred_trans = model(hand.to(device))
            loss = torch_loss(pred_rotate, pred_trans, grasp_rotate.to(device), grasp_trans.to(device))
            torch_optimizer.zero_grad()
            loss.backward()
            torch_optimizer.step()
            batch_loss = loss.item()
            loss_sum += batch_loss * len(hand)
            epoch_tqdm.set_description(f'batch: {batch_loss:.3f}, train: {train_loss:.3f}, val: {val_loss:.3f}')
        train_loss = loss_sum / train_len
        writer.add_scalar('Loss/Train', train_loss, e)
        model.eval()
        loss_sum = 0
        with torch.no_grad():
            for hand, grasp_rotate, grasp_trans in torch_val_iter:
                pred_rotate, pred_trans = model(hand.to(device))
                loss = torch_loss(pred_rotate, pred_trans, grasp_rotate.to(device), grasp_trans.to(device))
                batch_loss = loss.item()
                loss_sum += batch_loss * len(hand)
                epoch_tqdm.set_description(f'batch: {batch_loss:.3f}, train: {train_loss:.3f}, val: {val_loss:.3f}')
        val_loss = loss_sum / val_len
        epoch_tqdm.set_description(f'batch: {batch_loss:.3f}, train: {train_loss:.3f}, val: {val_loss:.3f}')
        writer.add_scalar('Loss/Val', val_loss, e)
        torch.save(model, args.save)
