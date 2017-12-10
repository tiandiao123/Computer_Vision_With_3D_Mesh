from tvtk.api import tvtk
from mayavi import mlab
import Image
import numpy as np

def update_view(scene, fx, fy, fz, x, y, z, vx = 0, vy = 1, vz = 0):
    """Set the view directly."""
    camera = scene.camera
    renderer = scene.renderer
    camera.focal_point = fx, fy, fz
    camera.position = x, y, z
    camera.view_up = vx, vy, vz
    scene.render()

def sphr2rgt(alpha = 0, beta = 0, gamma = 0):
	x = np.cos(gamma)*np.cos(alpha)
	y = np.cos(gamma)*np.sin(alpha)
	z = np.sin(gamma)
	return (x,y,z)

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
focal_length= 1
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

#pnts = []
labels_list = []
cnt = 0
n_samples = 32
with open('inpnt.txt') as file:
    lines = file.readlines()
    for pos in lines:
        pos = pos.replace('(','')
        pos = pos.replace(')','')
        pos = [float(item) for item in pos.split(', ')]
                #pnts.append(pos)
        for num in range(n_samples):
            angles = np.random.sample((3,))
            alpha = angles[0]*2*np.pi
            beta = angles[1]*np.pi
            gamma = angles[2]*2*np.pi
            add = sphr2rgt(alpha, beta, gamma)
            update_view(surf.scene,pos[0]-add[0],pos[1]-add[1],pos[2]-add[2],pos[0],pos[1],pos[2])
            mlab.move(forward=0, right=0, up=0)
            cnt = cnt+1
            mlab.savefig('dataset/'+str(cnt)+'.jpg',size=(320,320))
            pos.extend(angles)
            labels_list.append(pos)
np.save('labels',labels_list)
mlab.show()