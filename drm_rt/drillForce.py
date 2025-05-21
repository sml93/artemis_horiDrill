import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

gr = 1          ## gear ratio
dm1 = 0.02346     ## mm
dm2 = 0.01334
h1 = 0.0015     ## slope height @ 2 mm
h2 = 0.0015


def Force(fric, torque, h):
  f = fric
  tr = torque
  h = h
  F = []
  my_list = []
  for i in range(len(f)):
    for j in range(len(tr)):
      F.append(((2*gr*tr[j])/dm1 * ((np.pi*dm1 - f[i]*h)/(h + np.pi*dm1*f[i])) / 9.81))

  my_list = np.array_split(F, 10)
  # print(my_list)
  return F, f, tr, my_list


def main():
  f = np.linspace(0.3, 0.5, 50)
  tr = np.linspace(0.1, 0.6, 50)

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  X, Y = np.meshgrid(f, tr)
  Z1 = ((2*gr*Y/dm1 * ((np.pi*dm1 - X*h1)/(h1 + np.pi*dm1*X)) /9.81))
  Z2 = ((2*gr*Y/dm2 * ((np.pi*dm2 - X*h2)/(h2 + np.pi*dm2*X)) /9.81))

  print(Z1)
  print(Z2)

  Z_total = Z1 + Z2

  surf = ax.plot_surface(X, Y, Z_total, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  # ax.set_zlim(0, 40)
  ax.zaxis.set_major_locator(LinearLocator(10))
  # ax.zaxis.set_major_formatter('{x:.02f}')

  fig.colorbar(surf, shrink=0.5, aspect=5)
  ax.set_xlabel('fric')
  ax.set_ylabel('torque [N.m]')
  ax.set_zlabel('Force (kgf)')
  plt.show()



# def surfPlot():
#   fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#   f = np.linspace(0.3, 0.5, 10)
#   tr = np.linspace(0, 0.4, 10)
#   X, Y = np.meshgrid(f, tr)
#   Z = ((2*gr*Y/dm * ((np.pi*dm - X*h)/(h + np.pi*dm*X)) /9.81))

#   surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#   # ax.set_zlim(0, 40)
#   ax.zaxis.set_major_locator(LinearLocator(10))
#   # ax.zaxis.set_major_formatter('{x:.02f}')

#   fig.colorbar(surf, shrink=0.5, aspect=5)
#   ax.set_xlabel('fric')
#   ax.set_ylabel('torque [N.m]')
#   ax.set_zlabel('Force (kgf)')
#   plt.show()

# def main():
#   F, f, tr, my_list = Force()
#   fig = plt.figure()
#   ax = plt.axes(projection='3d')
#   xdata = tr
#   ydata = f
#   zdata = []
#   for i in range(len(my_list)):
#     zdata = my_list[i]
#     ax.plot3D(xdata, ydata, zdata)
#   ax.set_xlabel('fric')
#   ax.set_ylabel('torque [N.m]')
#   ax.set_zlabel('Force (kgf)')


#   # plt.show()


if __name__ == "__main__":
  main()
  # surfPlot()
