import numpy as np

class Polygon:
    def __init__(self, n_vertx, n_face, vertx, fcvertx, fcnorm, in_pnt):
        self.n_vertx = n_vertx  #number of vertices
        self.n_face = n_face  #number of faces(triangles)
        self.vertx = vertx  #coordinates of vertices
        self.fcvertx = fcvertx  #indices of vertices that form a triangle
        self.fcnorm = fcnorm  #norm vectors of faces
        self.in_pnt = in_pnt  #an inner point
    def detect(self, tst_pnt):
        cnt = 0
        v_ti = tst_pnt - self.in_pnt
        
        #avoid test_point == inner_point.
        if( np.linalg.norm(v_ti) <= np.finfo(float).eps ):
            return True
        
        #count the intersection points of the test_point-inner_point segment and all the faces of the polygon.
        for ind in range(n_face):
            p_tri = [vertx[i] for i in fcvertx[ind]]
            v_tri = np.array(p_tri)-np.array(p_inner)
            f_norm = fcnorm[ind]
            
            #determine whether the intersection point is on the segment.
            t_intrsec = np.inner((p_tri[0]-p_inner),f_norm)/np.inner(v_ti,f_norm)
            if( (t_intrsec > 1) | (t_intrsec < 0)):
                continue
                
            #determine whether the intersection point is in the triangle.
            prod = [np.inner(v_ti,np.cross(v_tri[j%3],v_tri[(j+1)%3])) for j in range(3)]
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


model.detect(p_inner)






















































#UNFINISHED!!!