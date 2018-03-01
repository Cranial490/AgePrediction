import os
from PIL import Image
import tensorflow as tf
import cv2 
import numpy as np
from random import shuffle
from numpy import genfromtxt
#add data path here to read data
dataPath = "data/"

dirs = os.listdir(dataPath)
feat = []
label = []
for clas in dirs:
	l = int(clas)
	featPath = dataPath + clas
	features = os.listdir(featPath)
	for img in features:
		feat.append(img)
	for i in range(len(features)):
		label.append(l)

if True:
	c = list(zip(feat, label))
	shuffle(c)
	feat,label = zip(*c)

train_feat = feat[0:int(0.6*len(feat))]
train_label = label[0:int(0.6*len(label))]
val_feat = feat[int(0.6*len(feat)):int(0.8*len(feat))]
val_label = label[int(0.6*len(feat)):int(0.8*len(feat))]
test_feat = feat[int(0.8*len(feat)):]
test_label = label[int(0.8*len(label)):]


def load_image(feat,label):
	addr = dataPath + str(label)+'/'+feat
	img = cv2.imread(addr)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	return img

def next_batch(feature,label,iteration,batch_size):
	batch = np.empty(227,227,batch_size)
	strt = iteration*batch_size
	end = strt+batch_size
	xbatch = feature[strt:end]
	ybatch = label[strt:end]
	for i in range(batch_size):
		img = load_image(xbatch[i],ybatch[i])
		batch[i] = img
	return batch,ybatch

x,y = next_batch(train_feat,train_label,0,60)

print x.shape


