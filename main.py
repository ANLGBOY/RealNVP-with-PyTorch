import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os

# --- configuration --- #
BATCH_SIZE = 128
LOG_INTERVAL = 50
EPOCHS = 20
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIM = 256
SAVE_PLT_INTERVAL = 5
N_COUPLE_LAYERS = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rcParams['figure.figsize'] = 8, 8
plt.ion()


# --- data loading --- #
train_data = datasets.make_moons(n_samples=50000, noise=.05)[0].astype(np.float32)
test_data = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)

# pin memory provides improved transfer speed
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# --- defines the model and the optimizer ---- #
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask):
        super().__init__()
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask

    def forward(self, x):
        x_m = x * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
        y = x_m + (1-self.mask)*(x*torch.exp(s_out)+t_out)
        log_det_jacobian = s_out.sum(dim=1)
        return y, log_det_jacobian

    def backward(self, y):
        y_m = y * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
        x = y_m + (1-self.mask)*(y-t_out)*torch.exp(-s_out)
        return x


class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers = 6):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'

        self.modules = []
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        for _ in range(n_layers-2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
        self.module_list = nn.ModuleList(self.modules)
        
    def forward(self, x):
        ldj_sum = 0 # sum of log determinant of jacobian
        for module in self.module_list:
            x, ldj= module(x)
            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z):
        for module in reversed(self.module_list):
            z = module.backward(z)
        return z


mask = torch.from_numpy(np.array([0, 1]).astype(np.float32))
model = RealNVP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, mask, N_COUPLE_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
prior_z = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))


# --- train and test --- #
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        z, log_det_j_sum = model(data)
        loss = -(prior_z.log_prob(z)+log_det_j_sum).mean()
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))


def test(epoch):
    model.eval()
    test_loss = 0
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            z, log_det_j_sum = model(data)
            cur_loss = -(prior_z.log_prob(z)+log_det_j_sum).mean().item()
            test_loss += cur_loss
            x_all = np.concatenate((x_all,data.numpy()))
            z_all = np.concatenate((z_all,z.numpy()))
        
        subfig_plot(1, x_all, -2, 3, -1, 1.5,'Input: x ~ p(x)', 'b')
        subfig_plot(2, z_all, -3, 3, -3,3,'Output: z = f(x)', 'b')

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


# --- etc. functions --- #
def sample(epoch):
    model.eval()
    with torch.no_grad():
        z = prior_z.sample((1000,))
        x = model.backward(z)
        z = z.numpy()
        x = x.numpy()

        subfig_plot(3, z, -3, 3, -3, 3, 'Input: z ~ p(z)', 'r')
        subfig_plot(4, x, -2, 3, -1, 1.5,'Output: x = g(z) (g: inverse of f)', 'r')

        if epoch % SAVE_PLT_INTERVAL == 0:
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig('results/'+'result_'+str(epoch)+'.png')


def subfig_plot(location, data, x_start, x_end, y_start, y_end, title, color):
        if location == 1:
            plt.clf()
        plt.subplot(2,2,location)
        plt.scatter(data[:, 0], data[:, 1], c=color, s=1)
        plt.xlim(x_start,x_end)
        plt.ylim(y_start,y_end)
        plt.title(title)
        plt.pause(1e-2)


# --- main function --- #
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        sample(epoch)