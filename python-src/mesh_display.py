import pymesh
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mayavi import mlab



mesh = pymesh.load_mesh("../PLY/target.ply")
print(mesh.vertices)
x=mesh.vertices[:,0]
y=mesh.vertices[:,1]
z=mesh.vertices[:,2]
s = mlab.mesh(x, y, z)
mlab.show()