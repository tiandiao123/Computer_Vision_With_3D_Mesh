import numpy as np
import itertools
import cudamat as cm
from scipy.spatial import distance
import codecs, json 
from osgeo import gdal
from tvtk.api import tvtk
from mayavi import mlab
import Image


class Polygon:
    def __init__(self, n_vertx, n_face, vertx, fcvertx, fcnorm, in_pnt):
        self.n_vertx = n_vertx  #number of vertices
        self.n_face = n_face  #number of faces(triangles)
        self.vertx = vertx  #coordinates of vertices
        self.fcvertx = fcvertx  #indices of vertices that form a triangle
        self.fcnorm = np.array(fcnorm)  #norm vectors of faces
        self.in_pnt = in_pnt  #an inner point
        self.p_tri = np.array([[vertx[fcvertx[ind][j]] for j in range(3) ] for ind in range(n_face)]) #vertices of triangles
        self.p_tri_slc = self.p_tri[:,0,:]
        self.v_tri = self.p_tri-np.array(in_pnt)
        prod_flat = np.array(((cm.CUDAMatrix(self.v_tri[:,0,:]).mult(cm.CUDAMatrix(self.fcnorm )) )).asarray())  #use GPU to calculate the element-wise product.
        self.prod1 = np.sum(prod_flat, axis = 1)  #v_tri dotprd fcnorm
        self.prod2 = np.array([[np.cross(self.v_tri[ind][j%3],self.v_tri[ind][(j+1)%3]) for j in range(3)] for ind in range(n_face)])  #v_inp2vtx crossprd
        #print self.prod2.shape
        
    def detect(self, tst_pnt):
        cnt = 0
        v_ti = tst_pnt - self.in_pnt
        
        #avoid test_point == inner_point.
        if( np.linalg.norm(v_ti) <= np.finfo(float).eps ):
            return True
        
        #count the intersection points of the test_point-inner_point segment and all the faces of the polygon.
        for ind in range(n_face):
            f_norm = fcnorm[ind]
            
            #determine whether the intersection point is on the segment.
            t_intrsec = self.prod1[ind]/np.inner(v_ti,f_norm)
            if( (t_intrsec > 1) | (t_intrsec < 0)):
                continue
                
            #determine whether the intersection point is in the triangle.
            prod = [np.inner(v_ti,self.prod2[ind][j]) for j in range(3)]
            if( ((prod[0] <= 0) & (prod[1] <= 0) & (prod[2] <= 0)) | ((prod[0] > 0) & (prod[1] > 0) & (prod[2] > 0)) ):
                cnt = cnt + 1 
        
        #iff the number of the intersection points is even, the test_point is an inner point.
        if( (cnt%2) == 0 ):
            return True
        else:
            return False

#read polygon information from file.
with open('PLY/ASR.ply') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
n_vertx = int((content[3].split())[2])
n_face = int((content[7].split())[2])
vertx = []
fcvertx = []
fcnorm = []
for i in range(14,14+n_vertx):
    vertx.append([float(j) for j in content[i].split()])
for i in range(14+n_vertx,14+n_vertx+n_face):
    fcvertx.append(([int(j) for j in content[i].split()])[1:4])
for i in range(14+n_vertx+n_face,14+n_face*2+n_vertx):
    fcnorm.append([float(j) for j in content[i].split()])

#find an inner point(verified).
p_tri = [vertx[i] for i in fcvertx[0]]
p_inner = np.mean(p_tri,0) - np.array(fcnorm[0])



#create a instance of class Polygon
model = Polygon(n_vertx, n_face, vertx, fcvertx, fcnorm, p_inner)

#sample a number of prod(n_sample) points within the region of model
max_vertx = np.amax(vertx,axis = 0)
min_vertx = np.amin(vertx,axis = 0)
n_sample = [10,10,5]
v_coor = [np.linspace(min_vertx[i],max_vertx[i],num = n_sample[i]) for i in range(3)]
mat_coor = [i for i in itertools.product(v_coor[0],v_coor[1],v_coor[2])]

#test points and add eligible candidates in mat_inner
mat_inner = []
for pnt in mat_coor:
    if( model.detect(pnt)):
        mat_inner.append(pnt)

# test detection points:
im1 = Image.open("sampled_texture.jpg")
im2 = im1.rotate(90)
im2.save("tmp/sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="tmp/sampled_texture.jpg")

my_texture=tvtk.Texture()
my_texture.interpolate=0
my_texture=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)


# surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))
surf=mlab.pipeline.surface(mat_inner)
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

poses=[]
focs=[]
extrinsic_matrices=[]

cam1,foc1=mlab.move()
poses.append(cam1)
focs.append(foc1)
focal_length=distance.euclidean(cam1,foc1)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]


poses=[]
focs=[]
extrinsic_matrices=[]
# Make an animation:
for i in range(36):
    # change distance
    delta = -i
    mlab.view(80,120,60+delta)
    #fig=mlab.figure(bgcolor=(1,1,1))
    
    cam1,foc1=mlab.move()
    poses.append(cam1)
    focs.append(foc1)
    matrix=surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)
    extrinsic_matrices.append(matrix)
    
    # Save the scene.
    surf.scene.save_png('test_images/anim%d.png'%i)