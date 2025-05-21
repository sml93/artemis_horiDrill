import numpy as np

from scipy.optimize import curve_fit


def linear(x, m, c):
  y = m*x + c
  return y

def linear_roots(y, m, c):
  x = (y-c)/m
  return x

def solveLinear(plot_xlist, plot_ylist):
  xdata = np.asarray(plot_xlist)
  ydata = np.asarray(plot_ylist)

  params, covariance = curve_fit(linear, xdata, ydata)

  fit_m = params[0]
  fit_c = params[1]

  # fit_y = linear(xdata, fit_m, fit_c)
  return fit_m, fit_c