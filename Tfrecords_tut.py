import numpy as np
import tensotflow as tf
tf_file_name = 'all.tfrecords'
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def CreateTfrecords(tf_file_name,dataPath,features,labels):
	#Opens the Tfrecord file 
	writer = tf.python_io.TFRecordWriter(tf_file_name)
	for i in range(len(features)):
		if not i%1000:
			print 'Train data: {}/{}'.format(i,len(features))
		# sys.stdout.flush()
		img = load_image(features[i],labels[i])
		label = labels[i]

	# 	Creating a feature for writing in the tfrecords file 
	feature = {'train/label':_int64_feature(label),'train/image':_bytes_feature(tf.compat.as_bytes(img.tostring()))}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	writer.write(example.SerializeToString())
	writer.close()
CreateTfrecords(tf_file_name,dataPath,feat,label)
