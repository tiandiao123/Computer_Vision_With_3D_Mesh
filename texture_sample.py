import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import scipy

img_eso=mpimg.imread('esophagus.jpg')
imgshape = img_eso.shape

sample_array=[]
r=50
center=[imgshape[0]/2,150]

img_crop = img_eso[center[0]-r:center[0]+r,center[1]-r:center[1]+r,:]
sample_array = np.reshape(img_crop,(-1,3))

length = 4*r**2
constructed_image=[]
for i in range(imgshape[0]):
    temp_array=[]
    for j in range(imgshape[1]):
        randnum=random.randint(0,length-1)
        temp_array.append(sample_array[randnum])
    constructed_image.append(temp_array)
constructed_image=np.array(constructed_image)

blur = cv2.GaussianBlur(constructed_image,(7,7),0)
dst = cv2.fastNlMeansDenoisingColored(blur,None,10,10,7,21)

#plt.imshow(blur)
#plt.show()

scipy.misc.imsave('sampled_texture.jpg', blur)
