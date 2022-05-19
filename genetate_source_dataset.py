import h5py
import numpy as np



f=h5py.File('./BO-28-28-100.h5','r')
data1=f['data'][:]
f.close()


f=h5py.File('./HS-28-28-100.h5','r')
data2=f['data'][:]
f.close()


f=h5py.File('./KSC-28-28-100.h5','r')
data3=f['data'][:]
f.close()


f=h5py.File('./CH-28-28-100.h5','r')
data4=f['data'][:]
f.close()


print(data1.shape) # (3600, 8100) 18
print(data2.shape) # (3200, 8100) 16
print(data3.shape) # (1800, 28900) 9
print(data4.shape)


data=np.vstack((data1,data2,data3,data4))
print(data.shape) # (8600, 28900) 8200 = 43*200
#data = data.reshape(800, 100, 28, 28, 3)

# indices = np.arange(data.shape[0])
# shuffled_indices = np.random.permutation(indices)
# data = data[shuffled_indices]
# print(data.shape)

f=h5py.File('./meta_train_' + str(data.shape[0]) + '_' + str(data.shape[1]) + '.h5','w')
f['data']=data
f.close()
