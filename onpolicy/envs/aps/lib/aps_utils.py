import torch
import numpy as np
import cvxpy as cp
from math import sqrt


def sinr_from_A(A, rho_d):
    A_diag = (np.abs(np.diag(A)))**2*rho_d
    A = rho_d*np.linalg.norm(A, axis=1, keepdims=False)**2
    return A_diag/(1+A-A_diag)

def set_random_seed(seed):
    # Set the seed for CPU and GPU (if using CUDA)
    torch.manual_seed(seed)
    # If using CUDA, set the seed for all GPUs as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_polar(a):
    assert isinstance(a, torch.Tensor)
    magnitude = torch.abs(a)
    magnitude = torch.clamp(magnitude, min=1e-20)
    phase = torch.angle(a)
    return magnitude, phase

def clip_abs(a):
    magnitude, phase = get_polar(a)
    return torch.polar(magnitude, phase)

# SOCP problem solver
# (G_dague, P_G, rho_d) set the problem's constraints
# t: is the currently computed lower bound sinr
def opti_OLP(t, G_dague, P_G, rho_d, M, K, mask=None):
    A = cp.Variable(shape=(K, K), complex=True)
    A_diag = cp.Variable(shape=(K, 1), pos=True)
    A_tilde = cp.Variable(shape=(K, K+1), complex=True)
    constraints = [cp.reshape(A_tilde[:, K], (K, 1))
                   # keep the last column constant
                   == np.ones((K, 1))/sqrt(rho_d)]
    U = cp.Variable(shape=(M, K), complex=True)
    for i in range(K):
        for j in range(K):
            # create the link between A and A_tilde for non diag element
            if i == j:
                constraints += [A_tilde[i, j] == 0]
                constraints += [A[i, j] == A_diag[i, 0]]
            # can't set A_tilde==A directly because diag(A_tilde)==0 but not
            # diag(A)
            else:
                constraints += [A_tilde[i, j] == A[i, j]]
        constraints += [A_diag[i, 0] >= sqrt(t)*cp.pnorm(A_tilde[i, :], 2)]

    Delta = G_dague @ A + P_G @ U

    # mask the unconnected channels
    if mask is not None:
        for m in range(M):
            for k in range(K):
                if not mask[m, k]:
                    constraints += [Delta[m, k] == 0]

    for m in range(M):
        constraints += [cp.pnorm(Delta[m, :], 2) <= 1]
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='MOSEK', verbose=False)

    return prob, A, U

def tpdv_parse(conf):
    if torch.cuda.is_available() and conf.simulation_scenario.if_use_cuda_sim:
        conf.simulation_scenario.device_sim = torch.device("cuda:0")
    else:
        conf.simulation_scenario.device_sim = torch.device("cpu")

    conf.simulation_scenario.float_dtype_sim = eval(conf.simulation_scenario.float_dtype_sim)

def get_adj(n_ues, n_aps, if_transpose=False):
    if if_transpose:
        same_ap_edges = []
        same_ue_edges = []  # edges id from 0 to n_ues*n_aps-1
        # UE type edges
        for k in range(n_ues):
            for m1 in range(n_aps):
                for m2 in range(m1+1, n_aps):
                    same_ue_edges.append([k*n_aps+m1, k*n_aps+m2])
                    # reverse to make graph unoriented
                    same_ue_edges.append([k*n_aps+m2, k*n_aps+m1])
        # AP type edges
        for m in range(n_aps):
            for k1 in range(n_ues):
                for k2 in range(k1+1, n_ues):
                    same_ap_edges.append([k1*n_aps+m, k2*n_aps+m])
                    # reverse to make graph unoriented
                    same_ap_edges.append([k2*n_aps+m, k1*n_aps+m])
        same_ue_edges, same_ap_edges = np.array(same_ue_edges), np.array(same_ap_edges)
    else:
        same_ap_edges = []
        same_ue_edges = []
        for cntr_1 in range(n_ues * n_aps):
            for cntr_2 in range(n_ues * n_aps):
                if cntr_1 == cntr_2:
                    continue
                if cntr_1 % n_ues == cntr_2 % n_ues:
                    same_ue_edges.append((cntr_1, cntr_2))
                elif int(cntr_1 / n_ues) == int(cntr_2 / n_ues):
                    same_ap_edges.append((cntr_1, cntr_2))
                else:
                    pass
        same_ue_edges, same_ap_edges = np.array(same_ue_edges).transpose(), np.array(same_ap_edges).transpose()

    return same_ue_edges, same_ap_edges
