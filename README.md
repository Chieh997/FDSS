# Finite Difference Spatial Spline (FDSS)

The Python implementation of the Finite Difference Spatial Spline (FDSS) model.

## Dependencies
This program requires the following Python packages:
+ Pytorch == 1.13.1
+ NumPy == 1.24.3
+ [torchmin](https://pytorch-minimize.readthedocs.io/en/latest/install.html) == 0.0.2 
+ [rpy2](https://rpy2.github.io/doc/latest/html/overview.html#installation) == 3.5.11
  
And the R package [mgcv](https://CRAN.R-project.org/package=mgcv) is required for mask matrices generation.

## Document
### Parameters
+ `gen_grid( *limit, r )`
    Generate grid for the FDSS model.
  + \*limit *(tuple of lists)*: several lists to define the limits of x and y 
  + r *(double)*: the resolution of the grid (rounded parameter $r$).

+ `gen_mask(grid, bnd)`
  Generate mask for the FDSS model.
  + grid *(tensor)*: X_grid, should be generated with `gen_grid` or `torch.meshgrid`
  + bnd *(r_list)*: 2D-boundary, can only accept r list.

> *class* FDSS(X_grid, *, mask = None,  device = "cpu")

The main object for the FDSS model.
+ `FDSS.__init__(X_grid, *, mask = None,  device = "cpu")`
  + X_grid *(tensor)*: 
    The grid $\mathbf{X}_r$ used to generate $\bar{\mathbf{Y}}$, should be generated with `gen_grid` or `torch.meshgrid`
  + mask *(bool_tensor)*:  
        Mask of grid $\mathbf{M}_r$ for irregular domain with the same size of X_grid, where `True`: interior of domain, `False`: exterior. (can be generated with `gen_mask`). 
  + device *(sting, {'cpu', 'cuda:0'})*  
+ `FDSS.fit(X, Y, sp=None, method='cg', **kwargs)`
  + X *(tensor)*: X samples.
  + Y *(tensor)*: Y samples. 
  + sp *(None or list)*: 
    Candidates of smoothing parameters for cross validation. If None, it will follow the automatic selection process.
  + method *(string, {"cg", "l-bfgs"})*: Gradient descent method in torchmin, default = 'cg'
  + **batch_num *(int)*: Number of batch to split the sample when calculating $\bar{\mathbf{Y}}$.
  + **print_every *(bool)*: Print out the result of each sp in the validation state or not. 
+ `FDSS.transform(X, Y=None, model=None, *, batch_num = 1, eval_func="mse")`
  + X *(tensor)*: X samples.
  + Y *(None or tensor)*: Real Y values, can be used to calculate MSE or ISE.
  + model *(None or FDSS.model)*: 
    FDSS model. If None, it will use the model that fitted with the `FDSS.fit` before.
  + batch_num *(int)*: Number of batch to split the sample when calculating $\hat{\mathbf{Y}}$.
  + eval_func *(string, {"mse", "ise"})*: Evaluation function. 