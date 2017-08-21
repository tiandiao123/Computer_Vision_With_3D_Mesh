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

### Strurure of The Neural Network:
* Firstly, please download the AlexNet pre-trained paramter using this [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy). 
* Then, we need to use AlexNet as our upper-layer neural network structure, and then Extend fully-connected neural network at the bottom to train our model. More details will be filled in. I implemented all the codes using tensorflow.  

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