import numpy as np


def evaluate_data(
    xi,
    kernel_evaluate,
    kernel_data,
    indices=None,
    axis=None,
    method="griddata",
    parallel=False,
    n_jobs=-1,
    backend="loky",
    **kwargs,
):
    r"""Evaluate the data at xi. Selected indices may be provided for a given
    axis.

    Parameters
    ----------
    xi : np.array
        Points at which to interpolate data.
    kernel_evaluate : Kernel.evaluate
        The Kernel evaluate method to be used.
    kernel_data : dict
        A dict with the Kernel data to be interpolated.
    indices : array_like or None, optional
        The indices of the values to extract. Also allow scalars for indices.
        Default is None.
    axis : int or None, optional
        The axis over which to select values. By default, the flattened input
        array is used. Default is None.
    method : str, optional
        Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
    parallel : bool, optional
        A flag to invoke joblib parallel execution.
    n_jobs : int, optional
        Number of jobs for joblib parallel execution. Default is -1.
    backend : str, optional
        Backend for joblib parallel execution. Default is ``"loky"``.
    **kwargs : dict
        Optional keyword-arguments are passed to the interpolation method.

    Returns
    -------
    dict
        A dict with the interpolated data.
    """

    if method == "griddata":
        from scipy.interpolate import griddata

        def upscale(points, values, xi, **kwargs):

            # griddata requires points and xi as 1d array for dim=1
            if len(points.shape) == 2 and points.shape[1] == 1:
                points = points.ravel()

            if len(xi.shape) == 2 and xi.shape[1] == 1:
                xi = xi.ravel()

            return griddata(points, values, xi, **kwargs)

    elif method == "rbf":
        from scipy.interpolate import RBFInterpolator

        def upscale(points, values, xi, **kwargs):

            # RBFInterpolator requires points and xi as 2d array, also for dim=1
            if len(points.shape) == 1:
                points = points.reshape(-1, 1)

            if len(xi.shape) == 1:
                xi = xi.reshape(-1, 1)
            return RBFInterpolator(y=points, d=values, **kwargs)(x=xi)

    else:
        raise ValueError("Method not supported.")

    out = dict()

    if not parallel:

        for label, kernel_parameters in kernel_data.items():
            out[label] = kernel_evaluate(
                xi=xi,
                upscale=upscale,
                kernel_parameters=kernel_parameters,
                indices=indices,
                axis=axis,
                **kwargs,
            )

    else:

        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(kernel_evaluate)(
                xi=xi,
                upscale=upscale,
                kernel_parameters=kernel_parameters,
                indices=indices,
                axis=axis,
                **kwargs,
            )
            for kernel_parameters in kernel_data.values()
        )

        out = dict(zip(kernel_data.keys(), results))

    return out
