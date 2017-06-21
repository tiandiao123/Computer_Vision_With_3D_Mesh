from osgeo import gdal
from tvtk.api import tvtk
from mayavi import mlab
import Image

im1 = Image.open("../texture.jpg")
im2 = im1.rotate(90)
im2.save("tmp/ortofoto90.jpg")
bmp1 = tvtk.JPEGReader(file_name="tmp/ortofoto90.jpg")

my_texture=tvtk.Texture()
my_texture.interpolate=0

my_texture=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)
surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/target.ply"))
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture
mlab.show()