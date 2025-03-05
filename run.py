import torch
import matplotlib.pyplot as plt

import model

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

if __name__ == "__main__":
    epochs = 20000
    n_ics = 10

    Nx = 100
    Nt = 100
    x_lin = torch.linspace(0,1,Nx)
    t_lin = torch.linspace(0,1,Nt)

    def g(x):
        return torch.sin(3 * torch.pi * x)

    ut = torch.stack([d_alembert(g, x_lin, t, 1.0) for t in t_lin])

    mlp = model.FCNN(200, 100, 4, 256)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-5)


    for i in range(epochs):
        loss = 0.0
        for k in range(n_ics):
            g = setup_g(torch.randn(1))
            ut = torch.stack([d_alembert(g, x_lin, t, 1.0) for t in t_lin])
            ut_hat = mlp(torch.cat([ut[:-2],ut[1:-1]], dim=1))
            loss += torch.mean((ut[2:]-ut_hat)**2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f'Epoch {i}, Loss {loss.item()}')

    # evaluate by rolling out from initial condition
    ut_hat = torch.zeros(Nt, Nx)
    ut_hat[:2] = ut[:2]
    for i in range(2, Nt):
        ut_hat[i, :] = mlp(torch.cat([ut_hat[i-2],ut_hat[i-1]], dim=0))

    for u, u_hat, t in zip(ut, ut_hat, t_lin):
        plt.plot(x_lin.numpy(), u.numpy(), c="k", alpha=0.2)
        plt.plot(x_lin.numpy(), u_hat.detach().numpy(), color=plt.cm.plasma(t / t_lin.max()), ls='--')
    plt.show()
    # model = Model()
    # data = Data()
    # train(model, data)

    # ...