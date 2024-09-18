import torch
from data.testFunction import eta
from rpy2.robjects import r as rcode
from rpy2.robjects.packages import importr
from numpy import array as nparray


baser = importr('base')
gen_datar = rcode('''
gen.data <- function(N,k=1,sde=.1){
    x1 <- runif(N)
    x2 <- runif(N)
    e <- rnorm(N, sd = .5)
    f1 <- sin(2*k*pi*x1)+cos(2*k*pi*x2)+2*sin(2*pi*(x1-x2))
    f <- x1+x2-1+f1/4
    df.sam <-data.frame(x1,x2, y=f+e, yt=f)
    return(df.sam)
}
''')
gen_gridr = rcode('''
function(r, k=1){
    x_lim <- c(0,1)
    y_lim <- c(0,1)
    xg <- seq(x_lim[1], x_lim[2], by=r)
    yg <- seq(y_lim[1], y_lim[2], by=r)
    gp <- expand.grid(list(x1 = xg, x2 = yg))
    f1 <- sin(2*k*pi*gp$x1)+cos(2*k*pi*gp$x2)+2*sin(2*pi*(gp$x1-gp$x2))
    gp$tru <- gp$x1+gp$x2-1+f1/4
    return(gp)
}
''')

def gen_eta_data(N, k,  *, engine = 'p',sde=.1, seed = None):
    assert engine in ['r', 'python', 'p'], "engine can only be 'r'(for r), 'p','python' (for python)"
    N, D = int(N), 2
    if engine == 'r':
        if seed:
            baser.set_seed(seed)
        data = nparray(gen_datar(N, k)).T # rpy2 work faster with numpy array than pytorch tensor
        data = torch.tensor(data)    
        X = data[:,:D]
        y =  data[:,D]
        yt = data[:,-1]
    else:
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        D = 2
        X = torch.rand(N, D)
        e = torch.normal(0, sde, size=(N,))
        yt = eta(X[:,0], X[:,1], k)
        y = yt+e
    return X, y, yt
    