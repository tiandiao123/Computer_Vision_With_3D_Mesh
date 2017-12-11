import os
import numpy as np
import shutil

new_dir = 'p_dataset/'

labels = np.load('labels.npy')

num_sample = labels.shape[0]

l_train = []
l_valid = []
l_test = []

r_partition = np.random.random_sample((20,))
print r_partition[2]
for ind in range(50):
	if(r_partition[ind] < .7):
		#l_train.append(labels[ind])
		for file in os.listdir('dataset/'): 
			old_path = 'dataset/'+str(ind+1)+'.jpg'
			new_path = new_dir+'train/'+str(ind+1)+'.jpg'
			shutil.copyfile(old_path,new_path)
