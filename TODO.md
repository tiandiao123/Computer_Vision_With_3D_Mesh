## Generate dataset

### From Scratch:
* Clone or download the clean branch of the repo.

* Run mesh_construct.py to make raw dataset.


You can see images are splitted in 3 partitions: train (70%), valid(ation) (20%) and test (10%) that are in /dataset/train/, /dataset/valid/ and /dataset/test/ respectively. The labels are in .npy format in /dataset/.

* Run write_binary.py to make a Tfrecord format dataset.

The Tfrecord format is relatively bigger in size but much faster to read when training and better supported in Tensorflow.
You can see them in /dataset/ after running write_binary.py

* Run train.py to train the neural network.

Under construction...

### Using dataset.zip:
* Clone or download the clean branch of the repo.

* unzip the dataset.zip in the root directory.

You can see images are splitted in 3 partitions: train (70%), valid(ation) (20%) and test (10%) that are in /dataset/train/, /dataset/valid/ and /dataset/test/ respectively. The labels are in .npy format in /dataset/.

* Run write_binary.py to make a Tfrecord format dataset.

The Tfrecord format is relatively bigger in size but much faster to read when training and better supported in Tensorflow.
You can see them in /dataset/ after running write_binary.py

* Run train.py to train the neural network.

Under construction...
