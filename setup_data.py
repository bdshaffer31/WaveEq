import torch

def wave_eq_solution(u0, t_lin, x_lin, c):
    """Analytical solution to the wave equation"""
    u = torch.zeros(len(t_lin), len(x_lin))
    u[0, :] = u0(x_lin)
    dx = x_lin[1] - x_lin[0]
    dt = t_lin[1] - t_lin[0]
    r = c * dt / dx
    