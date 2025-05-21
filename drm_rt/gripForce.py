import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

gr = 2         ## gear ratio
dm = 0.011     ## mm
h = 0.0015     ## slope height @ 2 mm


def Force():
  f = np.linspace(0.3, 0.5, 10)
  tr = np.linspace(0, 0.4, 10)
  F = []
  my_list = []
  for i in range(len(f)):         ## cannot do the friction like this, should do 1 friction for the whole range of torque. Right now it seems like with increased friction, the force increases.
    for j in range(len(tr)):
      F.append(((2*gr*tr[j])/dm * ((np.pi*dm - f[i]*h)/(h + np.pi*dm*f[i])) / 9.81))

  my_list = np.array_split(F, 10)
  print(my_list)

    # my_list.append(force)
  # print(my_list)
  # print(np.shape(my_list))
  # print(my_list)
  return F, f, tr, my_list

def surfPlot():
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

  f = np.linspace(0.3, 0.5, 10)
  tr = np.linspace(0, 0.4, 10)
  X, Y = np.meshgrid(f, tr)
  Z = ((2*gr*Y/dm * ((np.pi*dm - X*h)/(h + np.pi*dm*X)) /9.81))

  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

  # ax.set_zlim(0, 40)
  ax.zaxis.set_major_locator(LinearLocator(10))
  # ax.zaxis.set_major_formatter('{x:.02f}')

  fig.colorbar(surf, shrink=0.5, aspect=5)
  ax.set_xlabel('fric')
  ax.set_ylabel('torque [N.m]')
  ax.set_zlabel('Force (kgf)')
  plt.show()

def main():
  F, f, tr, my_list = Force()
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  xdata = tr
  ydata = f
  zdata = []
  for i in range(len(my_list)):
    zdata = my_list[i]
    ax.plot3D(xdata, ydata, zdata)
  ax.set_xlabel('fric')
  ax.set_ylabel('torque [N.m]')
  ax.set_zlabel('Force (kgf)')


  # plt.show()


if __name__ == "__main__":
  # main()
  surfPlot()
