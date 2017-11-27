from tvtk.api import tvtk
from mayavi import mlab
import Image
from scipy.spatial import distance
import numpy as np
import codecs, json 
from matplotlib.colors import LightSource
from mayavi.tools.engine_manager import get_engine

def update_view(scene, fx, fy, fz, x, y, z, vx = 0, vy = 1, vz = 0):
        """Set the view directly."""
        camera = scene.camera
        renderer = scene.renderer
        camera.focal_point = fx, fy, fz
        camera.position = x, y, z
        camera.view_up = vx, vy, vz
        scene.render()

addition = [[0,0,1],[0,1,0],[1,0,0],[0.7071,0.7071,0],[0.7071,0,0.7071],[0,0.7071,0.7071],[0.57735,0.57735,0.57735]]



# CONSTRUCT the 3D scene
#construct surface mesh
warp = mlab.pipeline.warp_scalar(mlab.pipeline.open("PLY/ASR.ply"), warp_scale=0)
normals = mlab.pipeline.poly_data_normals(warp)
normals.filter.feature_angle = 90
surf = mlab.pipeline.surface(normals)
surf.scene.point_smoothing = True
surf.scene.polygon_smoothing = True
surf.scene.line_smoothing = True

#apply texture to the mesh
im1 = Image.open("sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="sampled_texture.jpg")
textur=tvtk.Texture(input_connection=bmp1.output_port, interpolate=1)
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = textur
surf.actor.property.interpolation = 'phong'
surf.actor.property.specular = 0.1
surf.actor.property.specular_power = 5

#adjust the light source
surf.scene.light_manager.light_mode = "vtk"
surf.scene.light_manager.number_of_lights = 3

camera_lights = surf.scene.light_manager.lights
camera_lights[1].activate = False
camera_lights[2].activate = False

#change the background color to black
surf.scene.background = (0,0,0)

# SAVE images and camera poses
#calculate the intrinsic matrix
poses = []
focs = []
extrinsic_matrices = []
cam,foc=mlab.move()
poses.append(cam)
focs.append(foc)
focal_length=distance.euclidean(cam,foc)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

#pnts = []

cnt = 0
with open('inpnt.txt') as file:
    lines = file.readlines()
    for pos in lines:
        pos = pos.replace('(','')
        pos = pos.replace(')','')
        pos = [float(item) for item in pos.split(', ')]
        #pnts.append(pos)
        for add in addition:
            for d_1 in range(2):
                for d_2 in range(2):
                    for d_3 in range(2):
                        if( d_1 == 0 and d_2 == 0 and d_3 == 0):
                            update_view(surf.scene,pos[0]-add[0],pos[1]-add[1],pos[2]-add[2],pos[0],pos[1],pos[2])
                            cnt = cnt+1
                            surf.scene.save_png('dataset/'+str(cnt)+'.png')
                        if( d_1 == 1 and add[0] != 0):
                            add_0 = -add[0]
                            update_view(surf.scene,pos[0]-add[0],pos[1]-add[1],pos[2]-add[2],pos[0],pos[1],pos[2])
                            cnt = cnt+1
                            surf.scene.save_png('dataset/'+str(cnt)+'.png')
                            continue
                        if( d_2 == 1 and add[1] != 0):
                            add_0 = -add[1]
                            update_view(surf.scene,pos[0]-add[0],pos[1]-add[1],pos[2]-add[2],pos[0],pos[1],pos[2])
                            cnt = cnt+1
                            surf.scene.save_png('dataset/'+str(cnt)+'.png')
                            continue
                        if( d_3 == 1 and add[2] != 0):
                            add_0 = -add[2]
                            update_view(surf.scene,pos[0]-add[0],pos[1]-add[1],pos[2]-add[2],pos[0],pos[1],pos[2])
                            cnt = cnt+1
                            surf.scene.save_png('dataset/'+str(cnt)+'.png')
                            continue



#for i in range(10):
 #   mlab.move(0.2,0,0)
#cam,foc=mlab.move()
#mlab.orientation_axes()
#cam,foc=mlab.move()
#update_view(surf.scene, foc[0],foc[1],foc[2],cam[0], cam[1], cam[2])
#update_view(surf.scene, 1,0,0,0,0,0)
mlab.show(stop = True)