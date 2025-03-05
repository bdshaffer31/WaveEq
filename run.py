import torch

from model import Model
from setup_data import Data

def train(model, data):
    pass

if __name__ == "__main__":
    mesh = torch.linspace(0,1,100)
    model = Model()
    data = Data()
    train(model, data)

    # ...