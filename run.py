import torch
import matplotlib.pyplot as plt

import models

# from model import Model
# from setup_data import Data


def train(model, data):
    pass

def d_alembert(g, x_lin, t, c):
    return g(x_lin - c * t) + g(x_lin + c * t)

def setup_g(omega):
    def ic(x):
        return torch.sin(omega * torch.pi * 2 * x)
    return ic

def gen_soln(x_lin, t_lin, omega):
    g = setup_g(omega)
    return torch.stack([d_alembert(g, x_lin, t, 1.0) for t in t_lin])

def gen_train_dataset(n_ics, x_lin, t_lin):
    return torch.stack([gen_soln(x_lin, t_lin, i+1) for i in range(n_ics)])

def time_delay(ut):
    return torch.stack([ut[:-1],ut[1:]],dim=-1)

def fcnn_train():
    pass


if __name__ == "__main__":
    epochs = 5_000
    n_ics = 1

    Nx = 100
    Nt = 100
    x_lin = torch.linspace(0,1,Nx)
    t_lin = torch.linspace(0,1,Nt)

    # def g(x):
    #     return torch.sin(3 * torch.pi * x)

    # ut = torch.stack([d_alembert(g, x_lin, t, 1.0) for t in t_lin])

    training_data = gen_train_dataset(n_ics, x_lin, t_lin)

    # mlp = models.LipschitzFCNN(Nx * 2, 100, 4, 256)
    mlp = models.FCNN(Nx * 2, 100, 4, 256)
    # mlp = models.FourierFCNN(12, 4, 256)
    # model = models.TimeEmbedNN(mlp)
    model = models.fd_stencil(3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(epochs):
        loss = 0.0
        for k in range(n_ics):
            ut = training_data[k]
            # ut_hat = model(torch.stack([ut[:-2],ut[1:-1]],dim=1)) # stack time delayed series as batch dim
            # ut_hat = model(time_delay(ut[:-1])) # stack time delayed series as batch dim
            ut_hat = model(ut[:-1]) # stack time delayed series as batch dim
            # print(ut_hat.shape, ut[2:].shape)
            # loss += torch.mean((ut[2:]-ut_hat)**2)
            loss += torch.mean((ut[1:]-ut_hat)**2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f'Epoch {i}/{epochs}, Loss {loss.item():.4e}')

    # evaluate from each initial condition
    # ut_hat = model(time_delay(training_data[0][:-1]))
    ut_hat = model(training_data[0][:-1])
    for u, u_hat, t in zip(ut, ut_hat, t_lin):
        plt.plot(x_lin.numpy(), u.numpy(), c="k", alpha=0.2)
        plt.plot(x_lin.numpy(), u_hat.detach().numpy(), color=plt.cm.plasma(t / t_lin.max()), ls='--')
    plt.show()

    # evaluate by rolling out from initial condition
    ut_hat = torch.zeros(Nt, Nx)
    ut = training_data[0]
    ut_hat[:2] = ut[:2]
    for i in range(2, Nt):
        # ut_hat[i, :] = model(torch.stack([ut_hat[i-2],ut_hat[i-1]]))
        ut_hat[i, :] = model(ut_hat[i-1])

    for u, u_hat, t in zip(ut, ut_hat, t_lin):
        plt.plot(x_lin.numpy(), u.numpy(), c="k", alpha=0.2)
        plt.plot(x_lin.numpy(), u_hat.detach().numpy(), color=plt.cm.plasma(t / t_lin.max()), ls='--')
    plt.show()
    # model = Model()
    # data = Data()
    # train(model, data)

    # ...