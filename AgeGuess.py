import numpy as np 
import tensorflow as tf 
#Defining Helper Functions for CNN
def weights(shape):
	return tf.Variable(tf.random_normal(shape))
def biases(length):
	return tf.Variable(tf.constant(length,tf.float32))

def New_ConvLyr(input,num_channels,filter_size,num_filters,padding,strides,use_pooling=True,LRN_norm=False):
	shape = [filter_size,filter_size,num_channels,num_filters]
	weight = weights(shape)
	biase = biases(num_filters)
	layer = tf.nn.conv2d(input=input,filter=weigth,strides=strides,padding=padding)
	layer += biase
	layer = tf.nn.relu(layer)
	if use_pooling:
		tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	if LRN_norm:
		layer = tf.nn.local_response_normalization(layer,alpha=0.0001,beta=0.75)
	return layer
# num_elements ?? and get_shape()
def flatten_lyr(layer):
	layer_shape=layer.get_shape()
	num_features=layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer,[-1,num_features])
	return layer_flat

def new_fc_layer(input,num_inputs,num_outputs,use_relu=True,use_dropout=True):
	weight = weights(shape=[num_inputs,num_outputs])
	bias = biases(length=num_outputs)
	layer = tf.matmul(input,weight) + bias
	if use_relu:
		layer = tf.nn.relu(layer)
	if use_dropout:
		layer = tf.nn.dropout(layer,0.5)
	return layer
#Defining the hyperparameters for Convolution layers-----------------------------------------------
#Conv layer 1
filterSize1 = 7
filterNumber1 = 96
channelNumber1 = 3
strides1 = [1,4,4,1]
#Conv layer 2
filterSize2 = 5
filterNumber2 = 256
channelNumber2 = 96
strides2 = [1,1,1,1]
#Conv layer 3
filterSize3 = 3
filterNumber3 = 256
channelNumber3 = 256
strides3 =[1,1,1,1]
#For fully connected layers
numInputs1 = 256*7*7
fcSize6 = 512
fcSize7 = 512
outputSize = 8
#Defining Placeholders for holding data.
x = tf.placeholder(tf.float32,shape = [None,227,227,3])
y_true = tf.placeholder(tf.float32,shape = [None,8]) 
#-------------------Defining the Architecture of the Network-----------------------------------------
layerC_1 = New_ConvLyr(input=x,num_channels=channelNumber1,filter_size=filterSize1,num_filters=filterNumber1,padding='VALID',strides=strides1,use_pooling=True,LRN_norm=True)
layerC_1 = tf.pad(layerC_1,[[0,0],[2,2],[2,2],[0,0]])
layerC_2 = New_ConvLyr(input=layerC_1,num_channels=channelNumber2,filter_size=filterSize2,num_filters=filterNumber2,padding='VALID',strides=strides2,use_pooling=True,LRN_norm=True)
layerC_2 = tf.pad(layerC_2,[[0,0],[1,1],[1,1],[0,0]])
layerC_3 = New_ConvLyr(input=layerC_2,num_channels=channelNumber3,filter_size=filterSize3,num_filters=filterNumber2,padding='VALID',strides=strides3,use_pooling=True,LRN_norm=True)
layer_flatten = flatten_lyr(layerC_3)
fc6 = new_fc_layer(input=layer_flatten,num_inputs=numInputs1,num_outputs=fcSize6,use_relu=True,use_dropout=True)
fc7 = new_fc_layer(input=fc6,num_inputs=fcSize6,num_outputs=fcSize7,use_relu=True,use_dropout=True)
fc8 = new_fc_layer(input=fc7,num_inputs=fcSize7,num_outputs=outputSize,use_relu=False,use_dropout=False)
finalLyr = tf.nn.softmax(fc8)
yPredClass = tf.argmax(finalLyr,axis=1)
yTrueClass = tf.argmax(y_pred,axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc8,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(yPredClass,yTrueClass)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	trainBatchSize = 64
	BatchNumber = 400
	dataPath = 'train.tfrecords'
	feature = {'train/image': tf.FixedLenFeatures([],tf.string),
				'train/label': tf.FixedLenFeatures([],tf.int64)}
	filename_queue = tf.train.string_input_producer([dataPath],num_epochs=1)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features=feature)
	image = tf.decode_raw(features['train/image'],tf.float32)
	label = tf.cast(festures['train/label'],tf.int32)
	image = tf.reshape(image,[227,227,3])
	images.labels = tf.train_shuffle_batch([image, label],batch_size=trainBatchSize,capacity=50000,num_threads=1,min_after_dequeue=100)
	init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	for i in range(BatchNumber):
		xbatch,ybatch = sess.run([images,labels])
		ybatch=tf.one_hot(indices=ybatch, depth=10)
        ybatch= sess.run(ybatch)
        feed_dict_train = {x: xbatch,y_true: ybatch}
        sess.run(optimizer,feed_dict=feed_dict_train)
    #coord.request_stop()
    #coord.join(threads)
    #acc = sess.run(accuracy, feed_dict=feed_dict_train)
    #print acc
