# Finite Difference Spatial Spline (FDSS)

The Python implementation of the Finite Difference Spatial Spline (FDSS) model.

## Dependencies

This program requires the following: 
+ **Python**: Python version 3.7 or later 
+ **R**: R version 4.1 or later

#### Python packages:
+ Pytorch == 1.13.1
+ NumPy == 1.24.3
+ [torchmin](https://pytorch-minimize.readthedocs.io/en/latest/install.html) == 0.0.2 
+ [rpy2](https://rpy2.github.io/doc/latest/html/overview.html#installation) == 3.5.11
  
#### R package 
+ [mgcv](https://CRAN.R-project.org/package=mgcv): This R package is required for generating mask matrices. 

## Documentation
See [documentation](https://github.com/Chieh997/FDSS/blob/main/DOCUMENTAION.md) for details of model settings.

## Example
- [Example_IrrDom](https://github.com/Chieh997/FDSS/blob/main/Example_IrrDom.ipynb) provides an illustrative example of applying the FDSS method to irregular domains.
- [Example_RegDom](https://github.com/Chieh997/FDSS/blob/main/Example_RegDom.ipynb) provides an illustrative example of applying the FDSS method to regular domains.


## Basic Usage

```python
from fdss import FDSS, gen_grid, gen_mask

# generate grid and mask
X_grid = gen_grid(x1_lim, x2_lim, r)
mask = gen_mask(X_grid, boundary)               ## for irregular domains

# model initiation
model = FDSS(X_grid, device=device)             ## Regular Domain
model = FDSS(X_grid, mask=mask, device=device)  ## Irregular Domain

# model fit and predict
model.fit(X, y)
ypred, mse = model.transform(X, y, eval_func='mse')
```