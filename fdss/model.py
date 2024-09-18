import torch
import torch.nn.functional as F
import math
from torchmin import minimize as torch_minimize
import numpy as np

from rpy2.robjects import default_converter, numpy2ri, r as rcode
from rpy2.robjects.packages import importr

baser = importr('base')
def gen_grid(*limit, r):
    gx = []
    for l in limit:
        lmin, lmax = min(l), max(l)
        gx.append(torch.Tensor(np.array(rcode.seq(lmin, lmax, by=r))))
    grid =  torch.meshgrid(gx, indexing='ij')
    return torch.stack(grid, dim=-1)

def gen_mask(grid, bnd):
    '''
    grid : X_grid, should be generated with torch.meshgrid
    bnd: 2D-boundary, can only accept r list.
    '''
    grid_shape, d = grid.shape[:-1], grid.shape[-1]
    grid_np = grid.reshape(-1,d).cpu().numpy()
    with (default_converter + numpy2ri.converter).context():
        rcode.assign('gp', baser.as_data_frame(grid_np))
    rcode.assign('bnd', bnd)
    mask = rcode('''
        names(bnd) <- c('V1', 'V2')
        with(gp, mgcv::inSide(bnd, V1, V2))
    ''')
    mask = np.array(mask)
    return torch.tensor(mask).reshape(grid_shape)

'''
Ref: https://github.com/nacro711072/MFPCA/blob/master/lpr.py
'''
def Check_Bound(x, x0):
    d = x0.shape[-1]
    x_min_out = (x < x0.reshape(-1, d).min(0).values).any(1)
    x_max_out = (x > x0.reshape(-1, d).max(0).values).any(1)
    return(~(x_min_out | x_max_out))

def bin_data(x, y, x0, bin_weight = False, device = 'cpu'):
    grid_shape, d = torch.tensor(x0.shape[:-1], device=device), x0.shape[-1] # q,d
    ptp = lambda input, axis: input.max(axis, keepdim=True).values-input.min(axis, keepdim=True).values
    bin_width = ptp(x0.reshape(-1, d), 0)/(grid_shape - 1) # r
    boundary = Check_Bound(x, x0)
    x = x[boundary]
    y = y[boundary].squeeze()
    if bin_weight is True:
        bin_shape = grid_shape + torch.ones(d, dtype=torch.int, device=device)
        newx_int = ((x - x0.reshape(-1, d).min(0).values) / bin_width).int()
        newx_float = (x - x0.reshape(-1, d).min(0).values) / bin_width - newx_int
        bin_number = (torch.stack([newx_int + (i & (1<<torch.arange(d, device=device).flip(0))>0).int() for i in range(2**d)])*\
                    torch.cat((bin_shape.flip(0).cumprod(0).flip(0)[1:],torch.tensor([1], device=device)))).sum(2)
        w = torch.stack([1 - newx_float, newx_float])
        linear_w = w.select(2,0).T
        for i in range(1, d):
            linear_w = torch.einsum('ij,ki->ijk', linear_w, w.select(2,i)).reshape(-1, 2**(i+1))
        binx = torch.bincount(bin_number.reshape(-1), linear_w.T.reshape(-1), minlength = bin_shape.prod().item())
        sumy = torch.bincount(bin_number.reshape(-1), (linear_w.T * y).reshape(-1), minlength = bin_shape.prod().item())
        binx = binx.reshape(bin_shape.tolist()).cpu().numpy()
        sumy = sumy.reshape(bin_shape.tolist()).cpu().numpy()
        for i in range(bin_shape.size()[0]):
            binx = np.delete(binx, -1, i)
            sumy = np.delete(sumy, -1, i)
        binx =  torch.from_numpy(binx).to(device)
        sumy =  torch.from_numpy(sumy).to(device)
    else:
        position = torch.round(((x - x0.reshape(-1, d).min(0).values) / bin_width)).int()
        bin_number = (position * torch.cat((grid_shape.flip(0).cumprod(0).flip(0)[1:],torch.tensor([1], device=device)))).sum(1)
        binx = torch.bincount(bin_number, minlength=grid_shape.prod().item()).reshape(grid_shape.tolist())
        sumy = torch.bincount(bin_number, weights=y ,minlength= grid_shape.prod().item()).reshape(grid_shape.tolist())
    return (torch.stack((sumy, binx), dim=0), bin_width)

def pred_sparse_idx(x, x0, device = 'cpu'):
    N = x.shape[0]
    grid_shape, d = torch.tensor(x0.shape[:-1], device=device), x0.shape[-1] # q,d
    ptp = lambda input, axis: input.max(axis, keepdim=True).values-input.min(axis, keepdim=True).values
    bin_width = ptp(x0.reshape(-1, d), 0)/(grid_shape - 1) # r
    boundary = Check_Bound(x, x0)
    x = x[boundary]
    position = torch.round(((x - x0.reshape(-1, d).min(0).values) / bin_width))
    ## sparse matrix multiplication for prediction
    idx = (position * torch.cat((grid_shape.flip(0).cumprod(0).flip(0)[1:],torch.tensor([1], device=device)))).sum(1)
    indices = torch.vstack((torch.arange(N, device = device), idx))
    return torch.sparse_coo_tensor(indices, torch.ones(N, device=device), (N, grid_shape.prod()))

def toeplitz_torch(kernel, input_size, device="cpu"):
    '''
    Inspired by https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
    '''
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    val = []
    idx = []
    for i in range(k_h):
        for j in range(k_w):
            if kernel[i,j] != 0:
                val.append(kernel[i, j]*torch.ones(o_w, device=device))
                idx.append(torch.vstack((torch.arange(o_w,device=device), (i*i_w+j)+torch.arange(o_w,device=device))))

    val = torch.hstack(val)
    idx = torch.hstack(idx)
    B_h, B_w = torch.sparse_coo_tensor(idx, val, device=device).shape

    indices = []
    values = []
    for i in range(o_h):
        values.append(val)
        _idx = idx.clone()
        _idx[0] += i*o_w
        _idx[1] += i*i_w
        indices.append(_idx)

    values = torch.hstack(values)
    indices = torch.hstack(indices)

    return torch.sparse_coo_tensor(indices,values, device=device, size=(o_h*B_h, i_w*i_h))


class FDSS:
    def __init__(self, X_grid, *, mask = None,  device = "cpu") -> None:
        ''' 
        X_grid: tensor, with shape (n_x1, ..., n_xd, d). 
                The grid used to generate Y_bar, can be generated with torch.meshgrid. 
        mask: tensor, with shape (n_x1, ..., n_xd). 
              Mask of grid for irregular domain. True -> interior of domain, False -> exterior
        '''
        self._fitted = False
        self.device = torch.device(device)

        self.X_grid = X_grid.to(self.device)
        self.D = X_grid.shape[-1]
        assert 2*2 > self.D, "Technical restriction: 2m>d"

        _w_xx = torch.tensor([1,-2,1]).to(self.device) 
        self._w_xx = self.__weight(_w_xx, self.D)
        if self.D > 1:
            _w_xy = torch.tensor([[1,0,-1],[0,0,0],[-1,0,1]]).to(self.device) 
            self._w_xy = self.__weight(_w_xy, self.D)
        self.__ConFunc(self.D)

        self.mask = None
        if mask != None:
            assert mask.shape == self.X_grid.shape[:-1], 'Incorrect mask shape'
            self.mask = mask.to(self.device)
    
    def _kernel_matrix(self):
        if self.D == 2:
            K = toeplitz_torch(self._w_xx, self.X_grid.shape[:-1], device=self.device)
            K += toeplitz_torch(self._w_xx.T, self.X_grid.shape[:-1], device=self.device)
            K += 2*toeplitz_torch(self._w_xy, self.X_grid.shape[:-1], device=self.device)
            if self.mask != None:
                K = (K).mul(self.mask.flatten().to(self.device).expand_as(K))
            return K
        else: 
            print("Eigenvalues calculation only supports for 2D.") 
    
    def __eigenvalues(self, K=None):
        if K is None:
            K = self._kernel_matrix()
        KTK = (K).T.matmul(K) 
        e1, _ = torch.lobpcg(KTK, k=1, largest=True)
        en, _ = torch.lobpcg(KTK, k=1, largest=False)
        return torch.tensor([e1, en])
    
    def _sp_list(self, n=10, K=None, r=None, kappa=.01):
        eigens = self.__eigenvalues(K) if r is None else self.__eigenvalues(K)/r**2
        min = (kappa/((1-kappa)*eigens.max())).log()
        max = ((1-kappa)/(kappa*eigens.min())).log()
        return torch.logspace(min, max, n)

    def fit(self, X, Y, sp=None, method='cg', *, 
            batch_num = 1, bin_weight = False, n_sp=10,
            valid_size = .1, max_iter = 200, n_patient=3, valid_tol=1e-3, print_every=False,
            final_iter = None, tol=1e-4, return_all=False, reset_params = False):
        '''
        X: an n x d tensor
        Y: an n-vector (tensor)
        method: {"cg", "l-bfgs"}, gradient descent method in torchmin, default = 'cg'
        batch_num: number of batch for construct Y_bar (Note: NOT batch size)
        bin_weight: interpolate or not when construct Y_bar
        batch_num: number of batch used to calculate Bar Y_ij
        sp: list, candidates of smoothing parameters for cross validation.
        n_sp: number of smoothing parameters
        '''
        X = X.to(self.device)
        Y = Y.to(self.device)
        assert X.shape[0] == Y.shape[0]
        self.N, self.D = X.shape

        grid_shape = self.X_grid.shape[:-1]
        
        if self._fitted:
            params = self.Y_grid.nan_to_num()
        else: 
            params = torch.randn(grid_shape, device=self.device) 
        params = torch.nn.Parameter(params)

        if method != 'cg' or method != 'adam':
                method = 'l-bfgs'
        self._Min_Func(method)

        # Validation
        idx = torch.randint(int(self.N), (int(self.N*valid_size),))
        split_mask = torch.zeros(int(self.N)).bool().to(self.device)
        split_mask[idx] = True
        X_train, y_train = X[~split_mask], Y[~split_mask]
        X_valid, y_valid = X[split_mask], Y[split_mask]

        _Y_bin, self.r = self.y_bar_batch(X_train, y_train, batch_num = batch_num, bin_weight = bin_weight)
        self.r = self.r.float().squeeze()
        sumy_train, binx_train = _Y_bin
        _Y_bar_train = sumy_train/binx_train

        if sp is None:
            sp = self._sp_list(n=n_sp, K=self._kernel_matrix()/self.r[0])
        else:
            sp = torch.tensor(sp)
    
        mask_p, mask = self._penalty_mask(_Y_bar_train, grid_shape)

        mse_min = float('inf')
        last_mse = float('inf')
        mses = []
        mse_diffs=[]
        trigger_times = 0
        for lam in sp:
            penalty_func = lambda params : self._penalty_func(params, _Y_bar_train, binx_train, mask_p, lam)
            result = self.minimize(penalty_func, params, method=method, 
                        max_iter=max_iter, tol=tol, return_all=return_all)
            model = result if method == 'adam' else result.x
            if mask != None:
                model = torch.masked_fill(model, ~mask, float('nan'))
            _, mse = self.transform(X=X_valid, Y=y_valid, model=model)

            if print_every:
                print(f'lam = {lam:.2E}, mse = {mse:.5f} ({mse - mse_min:.3f}), iter = {result.nit}')
                mses.append(mse.item())
                mse_diffs.append((mse - mse_min).item())

            if (mse - mse_min) >= valid_tol and (mse - last_mse)>= (valid_tol):
                trigger_times += 1
                if trigger_times >= n_patient:
                    break
            else:
                trigger_times = 0
                if (mse - mse_min) < (valid_tol):
                    mse_min = mse
                    self.best_sp = lam
                    params = model.nan_to_num()
            last_mse = mse
        
        if print_every:
            self.logs = {
                'sp': sp.cpu(), 'mses': mses, 'mse_diff': mse_diffs
            }

        # Final Fit
        self._Y_bin, _ = self.y_bar_batch(X, Y, batch_num = batch_num, bin_weight = bin_weight)
        self.sumy, self.binx = self._Y_bin[0], self._Y_bin[1]
        self._Y_bar = self.sumy/self.binx

        if self.mask != None:
            mask_p, self.mask = self._penalty_mask(self._Y_bar, grid_shape)

        penalty_func = lambda params : self._penalty_func(params, self._Y_bar, self.binx, mask_p, self.best_sp)
        
        if reset_params: 
            params = torch.randn(grid_shape, device=self.device) 
            params = torch.nn.Parameter(params)
        else:
            final_iter = max_iter if final_iter is None else final_iter
        self.result = self.minimize(penalty_func, params, method=method,
                                    max_iter=final_iter, tol=tol, return_all=return_all)

        
        if method == 'adam':
            self.Y_grid = self.result
        else:
            self.Y_grid = self.result.x
            self.n_iter = self.result.nit

        if self.mask != None:
            self.Y_grid = torch.masked_fill(self.Y_grid, ~self.mask, float('nan'))

        self._fitted = True
    
    def _penalty_mask(self, Y_bar, grid_shape):
        if self.mask != None:
            mask_full = self.mask.logical_or(~Y_bar.isnan())
            if self.D == 2:
                mask = mask_full[1:-1, 1:-1]
            elif self.D == 3: 
                mask = mask_full[1:-1, 1:-1, 1:-1]
            else:
                print('Warning: not support for now.')
        else:
            mask_full = None
            mask = torch.ones((torch.tensor(grid_shape)-2).tolist()).bool().to(self.device)
        return mask, mask_full

    def _Min_Func(self, method):
        if method == 'adam':
            self.minimize = self._adam
        else:
            self.minimize = torch_minimize

    
    def transform(self, X, Y=None, model=None, *, batch_num = 1, eval_func="mse"):
        '''
        eval_func: {'mse', 'ise'}
        '''
        assert eval_func in ['mse','ise'], 'Choose a valid evaluation function: "mse", "ise"'
        if model is None:
            assert self._fitted
            model = self.Y_grid
        N, _ = X.shape
        batch = N//batch_num
        if batch > 1e7:
            batch = int(1e7) # something unknown will happen for sparse_coo_matrix if batch size > 1e7
            batch_num = math.ceil(N/batch)
        with torch.no_grad():
            Y_pred = torch.zeros((N,), device=self.device)
            for i in range(batch_num):
                if i < batch_num-1:
                    X_batch = X[batch*i:batch*(i+1)].to(self.device)
                else:
                    X_batch = X[batch*i:].to(self.device)
                sparse_idx = pred_sparse_idx(X_batch, self.X_grid, device=self.device)
                if i < batch_num-1:
                    Y_pred[batch*i:batch*(i+1)] = torch.sparse.mm(sparse_idx, model.reshape(-1,1)).squeeze()
                else:
                    Y_pred[batch*i:] = torch.sparse.mm(sparse_idx, model.reshape(-1,1)).squeeze()
            Y_pred = Y_pred.to(X.device)
            if Y is not None:
                if eval_func == "mse":
                    eval = (Y_pred - Y).square().nanmean()
                elif eval_func == "ise":
                    eval = (self.r.prod())*(Y_pred - Y).square().nansum()
                return Y_pred, eval
        return Y_pred
    
    def y_bar_batch(self, X, Y, batch_num, bin_weight = False):
        batch = self.N//batch_num
        for i in range(batch_num):
            # print('batch = ', i)
            if i < batch_num-1:
                X_batch, Y_batch = X[batch*i:batch*(i+1)].to(self.device), Y[batch*i:batch*(i+1)].to(self.device)
            else:
                X_batch, Y_batch = X[batch*i:].to(self.device), Y[batch*i:].to(self.device)
            ybin, bin_width = bin_data(X_batch, Y_batch, self.X_grid, bin_weight = bin_weight, device = self.device)
            if i > 0:
                Ybin += ybin
            else:
                Ybin = ybin
        return Ybin, bin_width

    def _penalty_func(self, y_pred, y, binx, mask, alpha):
        assert y_pred.shape == y.shape # binx
        return (y_pred-y).mul(binx).nan_to_num().norm() + alpha*self._roughness_penalty(y_pred, mask)
    
    def _roughness_penalty(self, f_ij, mask):
        D = self.D # self.m
        Df = 0
        f_ij = f_ij.unsqueeze(0)
        for d in range(D):
            dd1, dd2 = (d, D-1) if d < D-1 else (0,0)
            dr = self.r[d+1] if d < D-1 else self.r[0]
            _w_xx = self._w_xx.transpose(dd1,dd2).unsqueeze(0).unsqueeze(0)
            dxx = self.ConvND(f_ij, _w_xx/(dr**2)).squeeze()
            dxx = torch.masked_fill(dxx, ~mask, 0)
            Df += dxx.square().nansum()
        if self.D > 1:               
            for dd1 in range(D):
                for dd2 in range(D):
                    if dd1 == dd2:
                        continue
                    dr = self.r[[dd1, dd2]]
                    w = self._w_xy.transpose(dd1,dd2).unsqueeze(0).unsqueeze(0)
                    dxy = self.ConvND(f_ij, w/(4*dr.prod())).squeeze()
                    dxy = torch.masked_fill(dxy, ~mask, 0)
                    Df += dxy.square().nansum()
        return (self.r.prod())*Df

    
    def __weight(self, w, D):
        dw = len(w.size())
        padding = [0,0]*dw
        for _ in range(D-dw):
            w = w.unsqueeze(0)
            padding += [1,1]
        return F.pad(w, pad=padding)
    
    def __ConFunc(self, d):
        assert d > 0 and type(d) is int, 'd must be an positive integer'
        if d == 1:
            self.ConvND = F.conv1d
        elif d == 2:
            self.ConvND = F.conv2d
        elif d == 3:
            self.ConvND = F.conv3d
        else:
            print('Warning: Not support d > 3 for  now')