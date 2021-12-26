import numpy as np


def wasserstein_NMF_coefficient_step(X, K, D, gamma, rho, H, options):
    """
    :param X: the input data (n-by-m matrix)
    :param K: exp(-M/gamma) where M is the ground metric (n-by-t matrix).
            If K is symmetric, consider using options.Kmultiplication='symmetric'.
            If M is a squared Euclidean distance on a regular grid, consider using options.Kmultiplication='convolution'
             (see optional inputs for more information)
    :param D: the dictionary (t-by-k matrix)
    :param gamma: the regularization parameter of Wasserstein (scalar)
    :param rho: the regularization parameter for lambda (scalar)
    :param H: the initial dual variables (t-by-m matrix)
    :param options: an option structure:
%           * options.verbose: if true the function displays information at
%           each iteraion. Default false
%           * options.t0: the initial step size for gradient updates.
%           Default 1
%           * options.alpha: parameter for the backtracking line search.
%           Default 0
%           * options.beta: parameter for the backtracking line search.
%           Default .8
%           * options.dual_descent_stop: stopping criterion. Default 1e-5
%           * options.bigMatrices: if true some operations are made so that
%           less memory is used. It may affect efficiency. Default false.
%           * options.weights: weights on the wasserstein distance (1-by-m
%           matrix). Defaults to empty which is equivalent to ones(1,m).
%           * options.Kmultiplication: A string which defines how to
%           multiply by K or K':
%               + 'symmetric': K=K' so only multiply by K
%               + 'asymmetric': K~=K', if options.bigMatrices is false then
%               K' is precomputed
%               + 'convolutional': the ground metric is squared Euclidean
%               distance on a regular grid, so multiplication is by
%               exp(-gamma*M) is equivalent to a convolution with a
%               gaussian kernel, which can be computed dimensions one at a
%               time. K should then be a structure:
%                   o K.grid_dimensions: the dimensions of the original
%                   data
%                   o K.kernelSize: the size of the gaussian kernel
    :param isHist:
    :return: - lambda: the minimizer (k-by-m matrix)
%       - H: the dual optimal variable (t-by-m matrix)
%       - obj: the optimum (scalar)
%       - gradient: the gradient of the dual on H (t-by-m matrix)
    """
    options = checkOptionsWasserteinProjection(options)
    n = X.shape[0]
    if len(X.shape) == 1:
        m = 1
    else:
        m = X.shape[1]
    if options['Kmultiplication'] == 'symmetric':
        if not (K.shape[1] == n and abs(K - K.T).sum() / K.sum() < 1e-3):
            options['Kmultiplication'] = 'asymmetric'
    if options['Kmultiplication'] == 'asymmetric':
        n = K.shape[1]

    if not isHistogram(X):
        raise ValueError('Input vectors are not histograms. Check that they are non-negative, sum to one and do not '
                         'contain NaN of inf')

    multiplyK, multiplyKt = buildMultipliers(K, options)

    if 'pX' not in options or options['pX'] is None:
        pX = matrixEntropy(X)
    else:
        pX = options['pX']

    def entropyObj(H):
        expD = np.exp(-np.dot(D.T, H) / rho)
        if np.isnan(expD).any():
            expD[np.where(np.isnan(expD))] = expD[np.where(~np.isnan(expD))].max()
        sumE = expD.sum(0)
        obj = rho * np.log(sumE).sum()

        temp = np.dot(D, expD)
        for i in range(temp.shape[0]):
            try:
                temp[i, :] = temp[i, :] / sumE
            except IndexError:
                temp[i] = temp[i] / sumE
        grad = -temp
        return obj, grad

    if 'weights' not in options or options['weights'] == []:
        options['weights'] = np.ones(m)
    wassersteinObj = lambda H: computeWassersteinLegendreWeighted(X, H, gamma, pX, multiplyK, multiplyKt,
                                                                  options['weights'])

    def computeObj(H):
        obj, grad = wassersteinObj(H)
        objE, gradE = entropyObj(H)
        obj = obj + objE
        grad = grad + gradE
        gradNorm = np.linalg.norm(np.matrix(grad), ord='fro')
        grad = grad / gradNorm
        return obj, grad, gradNorm

    computeObjGrad = lambda G: computeObj(G)
    proximal = lambda G: G
    sumFunction = lambda x, t, y: x + t * y
    computeNorm = lambda matr: np.linalg.norm(np.matrix(matr), ord='fro')
    H, obj, gradient = accelerated_gradient(H, computeObjGrad, proximal, sumFunction, computeNorm,
                                            options['dual_descent_stop'], options['t0'], options['alpha'],
                                            options['beta'], options['verbose'])
    obj = -obj

    lambda0 = np.exp(-np.dot(D.T, H) / rho)
    lambda0_sum = lambda0.sum(0)
    for i in range(lambda0.shape[0]):
        try:
            lambda0[i, :] = lambda0[i, :] / lambda0_sum
        except IndexError:
            lambda0[i] = lambda0[i] / lambda0_sum

    return lambda0, H, obj, gradient


def checkOptionsWasserteinProjection(options):
    """
    Check the optional inputs of Wasserstein projection
    """
    if 't0' not in options or options['t0'] is None:
        options['t0'] = 1
    if 'alpha' not in options:
        options['alpha'] = 0.5
    if 'beta' not in options or options['beta'] is None:
        options['beta'] = 0.8
    if 'verbose' not in options or options['verbose'] is None:
        options['beta'] = 0
    if 'dual_descent_stop' not in options or options['dual_descent_stop'] is None:
        options['dual_descent_stop'] = 1e-5
    if 'bigMatrices' not in options or options['bigMatrices'] is None:
        options['bigMatrices'] = False
    if 'weights' not in options:
        options['weights'] = []
    return options


def isHistogram(X):
    # isHist = ~((np.isnan(X).any() or np.isinf(X).any()) and ~(X > 0).all() and ~(X.sum(axis=0) == 1).all())
    isHist = True
    if np.isnan(X).any() or np.isinf(X).any():
        isHist = False
    if not (X >= 0).all():
        isHist = False
    if not (X.sum(axis=0) - 1 < 1e-5).all():
        isHist = False
    return isHist


def buildMultipliers(K, options):
    if options['Kmultiplication'] == 'symmetric':
        multiplyK = lambda M: np.dot(K, M)
        multiplyKt = multiplyK
    elif options['Kmultiplication'] == 'asymmetric':
        multiplyK = lambda M: np.dot(K, M)
        multiplyKt = lambda M: np.dot(K.T, M)
    else:
        raise TypeError('Unknown multiplication type')
    return multiplyK, multiplyKt


def matrixEntropy(X):
    pX = X * np.log(X)
    pX[np.where(np.isnan(pX))] = 0
    pX = -pX.sum()
    return pX


def computeWassersteinLegendreWeighted(X, H, gamma, pX, multiplyK, multiplyKt, weights):
    H_weighted = np.ones(H.shape)
    for i in range(H.shape[0]):
        try:
            H_weighted[i, :] = H[i, :] / weights
        except IndexError:
            H_weighted[i] = H[i] / weights
    alphaTmp = np.exp(H_weighted / gamma)
    grad = multiplyK(alphaTmp)
    if (grad == 0).any():
        grad[grad == 0] = grad[grad > 0].min()
    if np.isinf(grad).any():
        grad[np.isinf(grad)] = grad[~np.isinf(grad)].max()
    obj = (((X * np.log(grad)).sum(0) * weights).sum() + pX) * gamma
    grad = alphaTmp * multiplyKt(X / grad)
    return obj, grad


def accelerated_gradient(H0, computeObjGrad, proximal, sumFunction, computeNorm, stop=1e-5, t0=1, alpha=0.5, beta=0.8,
                         verbose=0):
    def end_value(x):
        if type(x) == np.ndarray:
            if len(x.shape) == 1:
                return x[-1]
            else:
                return x[x.shape[0] - 1, x.shape[1] - 1]
        elif type(x) == list:
            return x[-1]
        else:
            return x

    H = H0
    objective, gradient, gradNorm = computeObjGrad(H)
    obj = objective
    checkObj = objective
    niter = 0
    t = t0
    numberEval = 1
    Glast = H
    Hlast = H
    if verbose:
        print('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, end_value(obj), t))
        print('\t\tGradnorm=%f stop=%f' % (gradNorm, stop))
    tol = stop * (1 + computeNorm(H))

    max_iter = 1000
    iter_step = 0

    while gradNorm > tol and iter_step < max_iter:
        iter_step += 1
        niter = niter + 1
        last = end_value(obj)
        prevt = t
        phi = lambda G: computeObjGrad(G)
        t, objective, H, grad, gradNorm = backtrack(phi, last, H, gradient, -(alpha * gradNorm), beta, t, sumFunction)
        numberEval = numberEval + np.log(t / prevt) / np.log(beta) + 1

        G = proximal(H)
        H = sumFunction(G, (niter - 2) / (niter + 1), sumFunction(G, -1, Glast))
        objective, gradient, gradNorm = computeObjGrad(H)

        obj = np.array([obj, objective])
        Glast = G
        numberEval = numberEval + 1

        tol = stop * (1 + computeNorm(H))

        if np.mod(niter - 1, 20) == 0:
            if verbose:
                print('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, end_value(obj), t))
                print('\t\tGradnorm=%f tol=%f' % (gradNorm, tol))
            Hlast = H

        if checkObj < end_value(obj):
            niter = 0
            H = Hlast
            objective, gradient, gradNorm = computeObjGrad(H)
            obj = np.array([obj, objective])
        checkObj = end_value(obj)

        t = min([t / np.sqrt(beta), t0])

    if verbose:
        print('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, end_value(obj), t))
        print('\t\tGradnorm=%f stop=%f' % (gradNorm, tol))

    return H, obj, gradient


def backtrack(phi, f, U, dir, alpha, beta, t, sumFunction):
    H = sumFunction(U, -t, dir)
    obj, grad, gradNorm = phi(H)
    objs = obj
    test = obj + gradNorm
    while ~np.isreal(obj) or ~np.isreal(test) or np.isnan(test) or np.isinf(test) or obj > f + alpha * t:
        t = beta * t
        H = sumFunction(U, -t, dir)
        obj, grad, gradNorm = phi(H)
        objs = [objs, obj]
        test = obj + gradNorm
    return t, obj, H, grad, gradNorm


if __name__ == '__main__':
    pass
