from tvtk.api import tvtk
from mayavi import mlab
import Image
from scipy.spatial import distance
import numpy as np
import codecs, json 
from matplotlib.colors import LightSource


im1 = Image.open("sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="sampled_texture.jpg")

textur=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = textur

mlab.show()