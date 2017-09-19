import numpy as np

def cg(trpo, g, inputs, sess=None, reg_coef=1e-5, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Implement conjugate gradient algorithm to compute the matrix iverse.
    It will compute A^-1*g by A*g with cg_iters iterations. If residual is lower than residual_tol, it will stop
    """
    sess = sess
    p = g.copy()
    r = g.copy()
    x = np.zeros_like(g)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print titlestr % ("iter", "residual norm", "soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        z = sess.run(trpo.fisher_prod_x_flatten, feed_dict = dict(inputs, **{trpo.xs_flatten:p})) + reg_coef * p
        v = rdotr / (p.dot(z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr % (i + 1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x
