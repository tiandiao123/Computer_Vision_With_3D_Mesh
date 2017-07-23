from tvtk.api import tvtk
from mayavi import mlab
import Image
from scipy.spatial import distance
import numpy as np
import codecs, json 
from matplotlib.colors import LightSource

# CONSTRUCT the 3D scene
#construct surface mesh
surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))

#apply texture to the mesh
im1 = Image.open("sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="sampled_texture.jpg")
textur=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = textur

#adjust the light source
surf.scene.light_manager.light_mode = "vtk"
surf.scene.light_manager.number_of_lights = 3
camera_lights = surf.scene.light_manager.lights
camera_lights[1].activate = False
camera_lights[2].activate = False

# SAVE images and camera poses
#calculate the intrinsic matrix
cam,foc=mlab.move()
poses.append(cam)
focs.append(foc)
focal_length=distance.euclidean(cam,foc)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

#move camera and save images(unfinished)
for i in range(50):
    #move camera by (10,0,0)
    mlab.move(10,0,0)

    #get new camera position and focal point
    cam,foc=mlab.move()
    poses.append(cam)
    focs.append(foc)
    
    #calculate the extrinsic matrix
    matrix=surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)
    extrinsic_matrices.append(matrix)
    
    #save the scene.
    surf.scene.save_png('saved_images/anim%d.png'%i)
    
#mlab.show()