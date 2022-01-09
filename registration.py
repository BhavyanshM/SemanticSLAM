import numpy as np


def align_icp(P, Q):
    print(P.shape, Q.shape)

    mean_p = np.mean(P, axis=0)
    mean_q = np.mean(Q, axis=0)

    W = Q.T @ P
    U, D, Vt = np.linalg.svd(W)

    R = U @ Vt
    t = mean_q - mean_p

    return R, t


def match_points(P, Q):
    Pm, Qm = [], []
    for p in P:
        for q in Q:
            if np.linalg.norm(p - q) < 1:
                Pm.append(p)
                Qm.append(q)
                # print("Match:", p, q)
    return np.vstack(Pm), np.vstack(Qm)
