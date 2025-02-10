from .ordering import *

def __col(Theta, s, nugget = 1e-12):
    r"""
    compute \Theta_{s, s}^{-1}e_j / \sqrt{e_j^T\Theta_{s, s}^{-1}e_j}

    A simple implementation is to invert \Theta_{s, s} directly using the following code

    m = torch.inverse(Theta[s][:, s])
    return m[:, -1] / torch.sqrt(m[-1, -1])

    However, it has some stability issues. When, \Theta_{s, s} is ill-conditioned, torch.inverse(Theta[s, s]) may not
    give us the accurate solution and e_j^T\Theta_{s, s}^{-1}e_j may be negative and torch.sqrt(m[-1, -1]) produces NaN value.

    Thus, instead, we use standard Cholesky directly on \Theta_{s, s}^{-1} and also adding a small nugget to prevent numerical issue
    """
    try:
        m = Theta[s][:, s] + nugget * torch.eye(len(s))
        L = torch.linalg.cholesky(m)
        ej = torch.zeros(len(s))  # Create a tensor of zeros
        ej[-1] = 1
        v = torch.cholesky_solve(ej.unsqueeze(-1), L, upper=False).squeeze(-1)
        # print(v.shape)
        if v[-1] < 0:
            raise ValueError(f"Negative value encountered for square root: {v[-1]}")
        vl = torch.sqrt(v[-1])
        return v / vl
    except torch.linalg.LinAlgError as e:
        # Handle Cholesky decomposition failure
        print(r"When doing sparse Cholesky decomposition, Cholesky decomposition of the submatrix \Theta_{s, s} " + f"failed: {e}")
        raise  # Optionally re-raise the exception after logging
    except ValueError as e:
        # Handle negative square root or other value issues
        print(r"When doing sparse Cholesky decomposition, \Theta_{s, s}[-1, -1] is negative. " + f"Value error encountered: {e}")
        raise  # Optionally re-raise the exception


def __cholesky(Theta, sparsity):
    n = Theta.size(0)
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    # Prepare storage for sparse matrix components
    data = torch.zeros(total_nonzeros, dtype=torch.float64)
    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        col_data = __col(Theta, s)
        start, end = indptr[i], indptr[i + 1]

        data[start:end] = col_data
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    sparse_cholesky = torch.sparse_coo_tensor(indices, data, size=(n, n))
    return sparse_cholesky

def non_zeros(n, sparsity):
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        start, end = indptr[i], indptr[i + 1]
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    return indices

def sparse_cholesky(Theta, Perm, sparsity) -> torch.sparse_coo_tensor:
    """
    >>> rho = 3
    >>> b = 5
    >>> n = 8
    >>> d = 3
    >>> rng = torch.Generator().manual_seed(7)
    >>> x = torch.rand(n,d,generator=rng) 
    >>> Perm,lengths = maximin(x) 
    >>> Perm
    tensor([0, 5, 3, 6, 7, 2, 4, 1])
    >>> lengths
    tensor([   inf, 0.8040, 0.6989, 0.4975, 0.4714, 0.3418, 0.3278, 0.2662])
    >>> L = torch.rand(b,n,n,generator=rng).tril()
    >>> Theta = torch.einsum("ijk,ilk->ijl",L,L) # for k in range(b): Theta[i] = L[i]@L[i].T
    >>> Theta[:,torch.arange(n),torch.arange(n)] = Theta[:,torch.arange(n),torch.arange(n)]+1e-3 # nugget term 
    >>> Theta.shape
    torch.Size([5, 8, 8])
    >>> Theta[0]
    tensor([[0.2724, 0.3264, 0.3858, 0.3839, 0.1393, 0.1722, 0.4031, 0.4833],
            [0.3264, 0.7175, 0.6080, 0.4806, 0.4515, 0.5028, 0.6134, 0.6339],
            [0.3858, 0.6080, 0.6677, 0.5753, 0.4978, 0.4262, 0.7193, 0.7949],
            [0.3839, 0.4806, 0.5753, 1.3623, 0.9304, 0.5336, 1.4865, 1.0918],
            [0.1393, 0.4515, 0.4978, 0.9304, 1.5913, 1.0055, 1.5668, 0.9922],
            [0.1722, 0.5028, 0.4262, 0.5336, 1.0055, 1.5723, 1.5460, 1.0418],
            [0.4031, 0.6134, 0.7193, 1.4865, 1.5668, 1.5460, 2.8253, 2.1963],
            [0.4833, 0.6339, 0.7949, 1.0918, 0.9922, 1.0418, 2.1963, 2.6564]])
    >>> sparsity = sparsity_pattern(x[Perm], lengths, rho)
    >>> sparse_cholesky(Theta[0], Perm, sparsity)
    tensor(indices=tensor([[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 4,
                            5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7],
                           [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                            5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]]),
           values=tensor([ 1.9161, -0.5225,  0.8266, -1.4690, -0.2276,  1.1444,
                           0.2649, -0.9883, -1.2064,  1.3920, -1.5508,  0.4956,
                           0.8363, -1.4666,  1.2563, -4.2426, -0.3803,  0.0358,
                          -0.0482,  3.1900,  8.2420, -0.2001, -1.3139, -0.6692,
                           0.2032, -4.8747,  2.4834,  5.3803, -0.7233, -1.0094,
                           0.1938,  0.1030, -5.9110,  1.2838,  3.1249]),
           size=(8, 8), nnz=35, dtype=torch.float64, layout=torch.sparse_coo)
    >>> sparse_cholesky(Theta[1], Perm, sparsity)
    tensor(indices=tensor([[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 4,
                            5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7],
                           [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                            5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]]),
           values=tensor([  1.8913,  -1.1955,   0.8992,  -0.9346,  -0.6380,
                            1.5252,   2.2416,  -0.9204,  -1.2729,   1.3058,
                           -1.9899,   0.5917,   1.5845,  -2.1288,   1.7328,
                            0.4113,   0.8822,  -3.1839,  -0.3592,   2.8538,
                            0.8421,   1.0987, -10.5841,   4.9679,  -7.9467,
                           -2.0699,   9.1711,   4.5694,  -1.2814,   2.7186,
                           -9.3972,  14.2291,  13.6930, -14.7421,   7.3095]),
           size=(8, 8), nnz=35, dtype=torch.float64, layout=torch.sparse_coo)
    """
    reordered_Theta = Theta[Perm][:, Perm]
    return __cholesky(reordered_Theta, sparsity)



