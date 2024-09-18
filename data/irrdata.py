import torch
from data.testFunction import FELSPLINE
from rpy2.robjects import r as rcode
from rpy2.robjects.packages import importr
from numpy import array as nparray

baser = importr('base')
mgcv = importr('mgcv')

gen_datar = rcode('''
function(n, sde){
    n_sam = 0
    fsb <- mgcv::fs.boundary()
    while(n_sam < n){
        x <- runif(2*n)*5-1; y<-runif(2*n)*2-1
        tru <- mgcv::fs.test(x,y,b=1)
        noise <- rnorm(2*n)*sde ## add noise
        df.all <- data.frame(x=x, y=y,value=tru+noise,tru=tru) 
        df.sam <- df.all[with(df.all, mgcv::inSide(fsb,x,y)),] ## remove outsiders
        df.sam <- na.omit(df.sam)
        n_sam <- nrow(df.sam)
        rm(x, y, tru, df.all)
    }
    return(df.sam[1:n, ])
}
''')
gen_gridr = rcode('''
function(r){
    x_lim <- c(-1,4)
    y_lim <- c(-1,1)
    fsb <- mgcv::fs.boundary()
    xg <- seq(x_lim[1], x_lim[2], by=r)
    yg <- seq(y_lim[1], y_lim[2], by=r)
    gp <- expand.grid(list(x = xg, y = yg))
    gp$tru <- mgcv::fs.test(gp$x, gp$y,b=1)
    # gp <- gp[with(gp, mgcv::inSide(fsb,x,y)),]
    # gp <- na.omit(gp)
    return(gp)
}
''')

fs  = FELSPLINE()
fsb = fs.boundary
xlim = [-1,4]
ylim = [-1,1]

def gen_fs_data(N, sde=.1, *, engine = 'p', seed = None):
    assert engine in ['r', 'python', 'p'], "engine can only be 'r'(for r), 'p','python' (for python)"
    N, D = int(N), 2
    if engine == 'r':
        if seed:
            baser.set_seed(seed)
        data = nparray(gen_datar(N, sde)).T # rpy2 work faster with numpy array than pytorch tensor
        data = torch.tensor(data)    
        X = data[:,:D]
        y =  data[:,D]
        yt = data[:,-1]
    else:
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        n_sam = 0
        while n_sam < N:
            x1 = torch.rand(2*N)*5-1
            x2 = torch.rand(2*N)*2-1
            X = torch.vstack([x1,x2]).T
            yt, mask =  fs.test(X[:,0], X[:,1])
            X = X[mask,:]
            yt = yt[mask]
            n_sam = len(X)
        X = X[:N,:]
        e = torch.normal(0, sde, size=(N,))
        yt = yt[:N]
        y = yt+e
    return X, y, yt
    