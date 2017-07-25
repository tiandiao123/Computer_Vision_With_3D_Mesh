import numpy as np
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


p_tri = [vertx[i] for i in fcvertx[0]]
p_inner = np.mean(p_tri,0) - np.array(fcnorm[0])
p_outer = np.mean(p_tri,0) + np.array(fcnorm[0])
v_norm = p_outer - p_inner
cnt = 0

for ind in range(n_face):
    p_tri = [vertx[i] for i in fcvertx[ind]]
    v_tri = np.array(p_tri)-np.array(p_inner)
    f_norm = fcnorm[ind]
    t_intrsec = np.inner((p_tri[0]-p_inner),f_norm)/np.inner(v_norm,f_norm)
    if( (t_intrsec > 1) | (t_intrsec < 0)):
        continue
    prod = [np.inner(v_norm,np.cross(v_tri[j%3],v_tri[(j+1)%3])) for j in range(3)]
    if( ((prod[0] <= 0) & (prod[1] <= 0) & (prod[2] <= 0)) | ((prod[0] > 0) & (prod[1] > 0) & (prod[2] > 0)) ):
        cnt = cnt + 1

#UNFINISHED!
