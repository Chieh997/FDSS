# Documentation

## Main Model

> class **FDSS**(X_grid, *, mask = None,  device = "cpu")

The main object for the FDSS model.

### Initialization
> FDSS.**\_\_init\_\_**(X_grid, *, mask = None,  device = "cpu")

Initial the FDSS model.
##### Arguments
  + `X_grid` *(tensor)*: 
    The grid $\mathbf{X}_r$ used to generate $\bar{\mathbf{Y}}$, should be generated with `gen_grid` or `torch.meshgrid`
  + `mask` *(bool_tensor)*: 
    Mask of grid $\mathbf{M}_r$ for irregular domains with the same size as `X_grid`. `True` for interior, `False` for exterior (can be generated with `gen_mask`).
  + `device` *(string, {'cpu', 'cuda:0'})*:
    Device for computation. Defaults to `"cpu"`.

### Fit the Model
> FDSS.**fit** (X, Y, sp=None, method='cg', **kwargs)

Fit the FDSS model with sample data.
##### Arguments
  + `X` *(tensor)*: Input samples (features).
  + `Y` *(tensor)*: Target samples (outputs).
  + `sp` *(None or list, optional)*: 
    Candidates for smoothing parameters for cross-validation. If `None`, automatic selection is used.
  + `method` *(string, {"cg", "l-bfgs"}, optional)*: 
    Gradient descent method in torchmin. Defaults to `'cg'`.
  + `**batch_num` *(int, optional)*: 
    Number of batches to split the sample when calculating $\bar{\mathbf{Y}}$.
  + `**print_every` *(bool, optional)*: 
    Whether to print the result of each smoothing parameter in the validation stage.

### Transform (Prediction)
> FDSS.**transform** (X, Y=None, model=None, *, batch_num = 1, eval_func="mse")

Predict data using the fitted FDSS model.
##### Arguments
  + `X` *(tensor)*: Input samples for prediction.
  + `Y` *(None or tensor, optional)*: 
    Real Y values, used for calculating evaluation metric. If set to `None`, no evaluation will be performed. Defaults to `None`.
  + `model` *(None or FDSS.model, optional)*: 
    A fitted FDSS model. If `None`, the model fitted with `FDSS.fit` will be used.
  + `batch_num` *(int, optional)*: 
    Number of batches to split the sample when calculating predictions $\hat{\mathbf{Y}}$.
  + `eval_func` *(string, {"mse", "ise"}, optional)*: 
    Evaluation function. Defaults to `"mse"`.

## Utility function
### Generate Grid
> **gen_grid**( *limit, r )

Generate a grid for the FDSS model.
##### Arguments
  + `*limit` *(tuple of lists)*: 
    Lists defining the x and y axis limits. 
  + `r` *(double)*: 
    Grid resolution (rounded parameter $r$).

### Generate Mask
>**gen_mask**(grid, bnd)
Generates a mask for the FDSS model, marking valid points in irregular domains.
##### Arguments
  + `grid` *(tensor)*: 
    The spatial grid, generated with `gen_grid` or `torch.meshgrid`
  + `bnd` *(r_list)*: 
    2D boundary definition , only r list is accepted.