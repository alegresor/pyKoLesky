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
        m = Theta[..., s, :][..., :, s] + nugget * torch.eye(len(s))
        L = torch.linalg.cholesky(m)
        ej = torch.zeros(L.size(-1))  # Create a tensor of zeros
        ej[-1] = 1
        v = torch.cholesky_solve(ej.unsqueeze(-1), L, upper=False).squeeze(-1)
        # print(v.shape)
        if (v[...,-1] < 0).any():
            raise ValueError(f"Negative value encountered for square root: {v[-1]}")
        vl = torch.sqrt(v[...,-1]).unsqueeze(-1)
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
    batch_shape = list(Theta.shape[:-2])

    n = Theta.size(-1)
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    # Prepare storage for sparse matrix components
    data = torch.zeros([total_nonzeros]+batch_shape, dtype=torch.float64)
    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    permute_dims = [-1]+[i for i in range(len(batch_shape))]

    for i in range(n):
        s = sorted(sparsity[i])
        col_data = __col(Theta, s)
        start, end = indptr[i], indptr[i + 1]

        data[start:end] = torch.permute(col_data,permute_dims)
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    sparse_cholesky = torch.sparse_coo_tensor(indices, data, size=[n, n]+batch_shape)
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
    # if Theta has shape (*bs,n,n) where bs is a list of batch size dimensions
    # then this method returns a torch.sparse_coo_tensor of shape (n,n,*bs) in order to efficiently store the sparse matrix 
    # call sparse_cholesky_dense in order to return a dense version of this sparse cholesky with the desired shape (*bs,n,n) 
    reordered_Theta = Theta[..., Perm, :][..., :, Perm]
    return __cholesky(reordered_Theta, sparsity)

def convert_sparse_U_to_batched_dense_U(U_sparse):
    return torch.permute(U_sparse.to_dense(),[i for i in range(2,U_sparse.ndim)]+[0,1])

def sparse_cholesky_dense(Theta, Perm, sparsity) -> torch.Tensor:
    """
    >>> rho = 2
    >>> bs = [4,6,5]
    >>> n = 8
    >>> d = 2
    >>> rng = torch.Generator().manual_seed(7)
    >>> x = torch.rand(n,d,generator=rng) 
    >>> Perm,lengths = maximin(x) 
    >>> Perm
    tensor([0, 4, 5, 2, 6, 1, 3, 7])
    >>> lengths
    tensor([   inf, 0.6741, 0.4758, 0.3775, 0.2253, 0.2226, 0.2063, 0.1839])
    >>> sparsity = sparsity_pattern(x[Perm], lengths, rho)
    >>> sparsity
    {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 3, 4], 5: [1, 2, 5], 6: [1, 3, 6], 7: [0, 3, 4, 5, 6, 7]}
    >>> L = torch.rand(bs+[n,n],generator=rng).tril()
    >>> Theta = torch.einsum("...jk,...lk->...jl",L,L)
    >>> Theta[...,torch.arange(n),torch.arange(n)] = Theta[...,torch.arange(n),torch.arange(n)]+1e-3 # nugget term 
    >>> Theta.shape
    torch.Size([4, 6, 5, 8, 8])
    >>> U = sparse_cholesky_dense(Theta, Perm, sparsity)
    >>> U.shape
    torch.Size([4, 6, 5, 8, 8])
    >>> U_flat = U.flatten(end_dim=-3)
    >>> U_flat.shape
    torch.Size([120, 8, 8])
    >>> Theta_flat = Theta.flatten(end_dim=-3)
    >>> assert Theta_flat.shape==U_flat.shape
    >>> for i in range(Theta_flat.size(0)):
    ...     U_i = sparse_cholesky_dense(Theta_flat[i], Perm, sparsity)
    ...     assert (U_i==U_flat[i]).all()
    """
    U_sparse = sparse_cholesky(Theta,Perm,sparsity)
    return convert_sparse_U_to_batched_dense_U(U_sparse)
