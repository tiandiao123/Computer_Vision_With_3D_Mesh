## Summer Research Project for Computer Vision

### Update Information:
* currently working on using extended AlexNet to train our model.  

### Description of the Research Project:
* construct a graphic platform for rendering 3D mesh using OpenGL, PyMesh, mlab, etc.
* Texture 3D-Mesh Object using mayavi library and OPENGL
* Sampled images' pixels, and appying texture mapping to all the 3D mesh model to make them become more realistic
* change views of 3D mesh in different ways, and collect data for training of the neural work
* extracting camera extrinsic and intrinsic matrix for deep learning training
* Using Deep Learning to study target mesh location based on tag and large data sets of images

### How to generate more data:
* you can just use the following code, and adjust the number of iterations of for loop to generate more pictures, and the pictures' views will depend on your original camera' location and moving style you will set:
```
for i in range(200):
    # change distance
    delta = -(float(i)/float(200))*5
    mlab.view(80,130,45+delta)
    mlab.roll(30+delta)
    mlab.yaw(20-2*delta)
    mlab.pitch(20+2*delta)
    #surf.scene.camera.roll(10+delta)
    #surf.scene.camera.azimuth(10-delta)
    #fig=mlab.figure(bgcolor=(1,1,1))
    
    cam1,foc1=mlab.move()
    roll = mlab.roll()
    #azui = mlab.view()[0]
    #elevation = mlab.view()[1]
    yaw = 20-delta*2
    pitch = 20+delta*2
    temp = list(cam1)+[roll,yaw,pitch]
    #print(temp)
    poses.append(temp)
    surf.scene.save_png('test_images/anim%d.png'%i)
```


### Strurure of The Neural Network:
* Firstly, please download the AlexNet pre-trained parameter using this [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy). 
* Then, we need to use AlexNet as our upper-layer neural network structure, and then Extend fully-connected neural network at the bottom to train our model. More details will be filled in. I implemented all the codes using tensorflow.  
* The neural network improve the result of error between actual cameras' parameters and predicted cameras' parameters(cameras' locations, cameras' pitch, yall, roll) into less than 5. 
* When training, you can just try a small data set to see how it will work in your system(for example, I'd like try 200-800 pictures to reduce the training time, after that I will use large data set 10,000-80,000 pictures to train my neural network)!

### Installation:
* [OPENGL](https://www.opengl.org/) for C++ codes to run
* [VTK](http://www.vtk.org/download/) library for handling mesh object
* [mayavi](http://docs.enthought.com/mayavi/mayavi/) library for ploting mesh object

* Here are some easy command lines for installing VTK and mayavi in conda environment:
```
conda install -c anaconda vtk=6.3.0
conda install -c anaconda mayavi=4.5.0
```
### Testing:
* For running C++ files, please check this [website](http://www.vtk.org/Wiki/VTK/Examples/Cxx) to change VTK path to run C++ files since it depends which location your VTK is in.
* For those are not familiar with CMAKE and VTK, I recommend that you can stick to Python codes. 

More details will be filled in!