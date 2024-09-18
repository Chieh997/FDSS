import torch
import math

from matplotlib import colormaps
import matplotlib.pyplot as plt

#### Irregular domain
class FELSPLINE:
    def __init__(self, r0=.1,r=.5, l=3, n_theta=20):
        self.r0, self.r, self.l =  r0, r, l
        rr = r+(r-r0)
        theta = torch.linspace(math.pi, math.pi/2, steps=n_theta)
        x, y = theta.cos()*rr, theta.sin()*rr

        theta = torch.linspace(math.pi/2, -math.pi/2, steps=2*n_theta)
        x, y = torch.cat((x, theta.cos()*(r-r0)+l)), torch.cat((y, theta.sin()*(r-r0)+r))

        theta = torch.linspace(math.pi/2, math.pi, steps=n_theta)
        x, y = torch.cat((x, theta.cos()*r0)), torch.cat((y, theta.sin()*r0))

        x, y = torch.cat((x, x.flip(dims=(0,)))), torch.cat((y, -y.flip(dims=(0,))))
        self.boundary = torch.vstack((x, y)).T

    def test(self, x,y, b=1,exclude=True):
        r0, r, l = self.r0, self.r, self.l
        q = math.pi*r/2
        a = torch.zeros_like(x)
        d = torch.zeros_like(x)
        idx = (x >= 0) & (y>0)
        a[idx] = q + x[idx]
        d[idx] = y[idx] - r

        idx = (x >= 0) & (y<=0)
        a[idx] = -q - x[idx]
        d[idx] = -y[idx] - r

        idx = x < 0 
        a[idx] = -(y[idx]/x[idx]).atan()*r
        d[idx] = (x[idx]**2+y[idx]**2).sqrt() - r

        f = a*b+d**2 # the original

        ## create exclusion index
        idx = (d.abs() > r-r0) | ((x>l) & ((x-l)**2+d**2 > (r-r0)**2))
        if (exclude): 
            f[idx] = float('nan')

        return f, ~idx
    
    def plot(self, x1, x2, z, pred = False, *, linew = 1.5,level = 20, show_label = True):
        assert (x1.shape == x2.shape) and (x1.shape == z.shape), 'x1, x2, z must have same shape for contour plot'
        fsb = self.boundary
        xlim = [x1.min().item(),x1.max().item()]
        ylim = [x2.min().item(),x2.max().item()]
        # x, y = torch.meshgrid(xm, yn, indexing='xy')
        # tru, _ = self.test(x,y)
        zimg = z.T if pred else z
        plt.imshow(zimg.cpu(), interpolation='bilinear', cmap=colormaps['RdYlBu'], extent=xlim+ylim)
        plt.plot(fsb[:,0], fsb[:,1], color='black')
        cs = plt.contour(x1,x2, z.cpu(), linewidths = linew, levels = level, colors='k')
        if show_label:
            plt.clabel(cs, inline=True, fontsize=10)
        # plt.show()

#### Regular domain
import math
def eta(x1, x2, k):
    f = torch.sin(2*k*math.pi*x1)+torch.cos(2*k*math.pi*x2)+2*torch.sin(2*math.pi*(x1-x2))
    return x1+x2-1+f/4
