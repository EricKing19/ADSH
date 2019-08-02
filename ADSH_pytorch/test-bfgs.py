from scipy.optimize import fmin_l_bfgs_b, fmin, minimize
import numpy as np

# def myfunc(x):
#     return x ** 2 - 4 * x + 8
#
# x0 = np.ones((1))
# xopt = fmin(myfunc, x0)
#
# print (xopt)


def rosen(x,a):
    """The Rosenbrock function"""
    x = np.reshape(x,(2,2))
    temp = np.dot(x.transpose(),a)
    temp2 = np.dot(temp,x)
    loss = np.trace(temp2)
    return loss

def func_deriv(x,a):
    """The Rosenbrock function"""
    x = np.reshape(x, (2, 2))
    grad = (a + a.transpose()).dot(x)

    return grad.flatten()
x0 = np.ones((2,2))
x = np.ones((2,2))
a = np.reshape(x,(1,-1))
res = minimize(rosen, x0, args=(x), method='L-BFGS-B', jac=func_deriv, tol=1e-8, options={'disp': True})

print(res.x)