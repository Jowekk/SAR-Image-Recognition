import cv2
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

width = 750
height = 1024
channels = 6 
datafile ='../data/data.mat'
labelfile = '../data/Label_Flevoland_15cls.mat'
data_temp = scio.loadmat(datafile)
label_temp = scio.loadmat(labelfile)
labels = label_temp['label']
#thresh_labels = cv2.threshold(labels, 0.5, 1,cv2.THRESH_BINARY)
imgs = np.zeros((width,height,channels), dtype=np.float32) #TODO padding
imgs[:,:,0] = data_temp['A']
imgs[:,:,1] = data_temp['B']
imgs[:,:,2] = data_temp['C']
imgs[:,:,3] = data_temp['D']
imgs[:,:,4] = data_temp['E']
imgs[:,:,5] = data_temp['F']

#flatt = labels.flatten()
#print np.unique(flatt)
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
n = 8
nums = 13960
temp_num = 0
num_classes = 16
X = np.zeros((nums, n, n, channels), dtype=np.float32)
Y = np.zeros((nums, 1) ,dtype=int)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
thresh_labels = cv2.erode(labels, kernel)
for i in range(width):
    for j in range(height):
        if thresh_labels[i,j] != 0  and random.random() < 0.1 :
            if (i-4)>0 and (i+4)<width and (j-4)>0 and (j+4)<height and temp_num < nums:
                X[temp_num, :, :, :] = imgs[i-4 :i+4, j-4 :j+4, :]
                Y[temp_num] = labels[i,j]
                temp_num = temp_num + 1
                #print "Number of samples:", temp_num
                
temp_data = list(zip(X, Y))
random.shuffle(temp_data)
X[:,:,:,:], Y[:] = zip(*temp_data)

one_hot = np.zeros((nums, num_classes))
for num in range(nums):
    one_hot[num, Y[num]] = 1

print(X.shape)
print(one_hot.shape)   
            

'''
n = 8
num_classes = 16
it_h = int(np.floor(height/n))
it_w = int(np.floor(width/n))
nums = int(it_w * it_h)
X = np.zeros((nums, n, n, channels), dtype=np.float32)
Y = np.zeros((nums, 1) ,dtype=int)
iteration = 0
for i in range(it_w):
    for j in range(it_h):
        accumulator = np.zeros((num_classes, 1))
        part_img = labels[i*n :(i+1)*n, j*n :(j+1)*n]
        for w_n in range(n):
            for h_n in range(n):
                accumulator[part_img[w_n,h_n]] = accumulator[part_img[w_n,h_n]] + 1
        Y[iteration] = np.where(accumulator == np.max(accumulator))[0][0]
        X[iteration,:,:,:] = imgs[i*n :(i+1)*n, j*n :(j+1)*n, :]
        iteration = iteration + 1

temp_data = list(zip(X, Y))
random.shuffle(temp_data)
X[:,:,:,:], Y[:] = zip(*temp_data)

one_hot = np.zeros((nums, num_classes))
for num in range(nums):
    one_hot[num, Y[num]] = 1

print(X.shape)
print(one_hot.shape)
'''


        





