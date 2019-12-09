import tensorflow as tf 
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.models import Sequential, Model 
from keras.layers.core import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import regularizers
from keras import layers
from tensorflow.python.framework import ops 
from sklearn.metrics import accuracy_score, confusion_matrix
import h5py
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, UpSampling2D, AveragePooling2D, BatchNormalization
import sys
from scipy.misc import imread, imresize
import itertools

def residual_pool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use 1*1 conv to match the dimension of channels of previous block.
	then have residual block
	then have a pooling layer
	BN: batch normalization (true or false)
	"""
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:
		identity = BatchNormalization(axis=-1)(identity)
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:
		z = BatchNormalization(axis=-1)(z)	
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		z = layers.add([a, b])
	else:
		z = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)	
	return z

def residual_pool(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	build residual block
	then have a pooling layer
	"""
	identity = prev_layer
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		z = layers.add([a, b])
	else:
		z = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
	return z

def residual_nopool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use 1*1 conv to match the dimension of channels of previous block.
	then have residual block
	"""
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:	
		identity = BatchNormalization(axis=-1)(identity)
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)	
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def residual_block(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	build residual block
	"""
	identity = prev_layer
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def residual_convpool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use average pooling to match feature map size
	use 1*1 conv to match the dimension of channels.
	use conv with strides 2 to replace pooling	
	pad: when the size is even use 'same', otherwise 'valid'
	"""
	# identity = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(prev_layer)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
		identity = layers.add([a, b])
	else:
		identity = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(identity)
	if BN:	
		identity = BatchNormalization(axis=-1)(identity)
	z = Conv2D(num_filter, (3, 3), strides=(2,2), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = Activation(activation)(z)
	for i in range(num_layers-2):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def image_preprocessing(file, dataset, labelset, label, process_index):
	# p_basis = PrimitiveBasis(n_states=2)
	first_index = int(process_index[0])
	second_index = int(process_index[1])
	third_index = int(process_index[2])
	short_dim = [256, 288, 320, 352]
	data = imread(file)
	dim = short_dim[first_index]
	data_temp = imresize(data, (dim,dim*3))
	crop_range = [0,1,2]
	i = crop_range[second_index]	
	square = data_temp[:, (dim*i):(dim*(i+1))]
	if third_index == 0:
		crop5 = square[(dim/2-112):(dim/2+112),(dim/2-112):(dim/2+112)]
		dataset.append(crop5)
		labelset.append(label)
	else:
		crop6 = imresize(square, (224,224))
		dataset.append(crop6)
		labelset.append(label)
	 

def generator(files, labels, shuffle, batch):
	path = '/raid/zyz293/NIST_data/'
	while 1:
		index = np.arange(len(files))
		if shuffle:
			np.random.shuffle(index)
		for i in range(len(files) / batch):
			x = []
			y = []
			file_list = files[index[i*batch:(i+1)*batch]]
			label_list = labels[index[i*batch:(i+1)*batch]]
			for j in range(len(file_list)):
				file_temp = file_list[j].split('&')[0]
				process_temp = file_list[j].split('&')[1]
				temp = path + file_temp[:-3] + 'png'
				image_preprocessing(temp, x, y, label_list[j], process_temp)
			x = np.array(x)
			x = np.expand_dims(x, axis=-1)
			# print np.array(x).shape, np.array(y).shape
			yield x, np.array(y)

## parameter sets
crop_size = 24
batchsize = 3 * crop_size
n_epoch = 2
patience = 20
L2 = 0.00
# lr = 0.001
bn = True
activation = 'relu'
pool = 0 # 0 for maxpooling, 1 for sum of max and average pooling
inp_size = (224, 224, 1) # input shape
# create 2D CNN model
print 'create model'
def build_model():
	inp = Input(shape=inp_size)
	# x = Conv2D(64, (7, 7), strides=(2,2), padding='same', W_regularizer=l2(L2))(inp)
	x = Conv2D(16, (3, 3), padding='same', W_regularizer=l2(L2))(inp)
	if bn:
		x = BatchNormalization(axis=-1)(x)
	x = Activation(activation)(x)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		x = layers.add([a, b])
	else:
		x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

# residual_pool_changeChannelnum: change # of channels of prev_layer and have pool at end
# residual_pool: just have pool at end
# residual_nopool_changeChannelnum: change # of channels of prev_layer and no pool
# residual_block: regular residual block
# residual_convpool_changeChannelnum: change # of channels of prev_layer and use conv to replace pool

	x = residual_nopool_changeChannelnum(32, 3, x, L2, bn, activation, pool)
	x = residual_pool(32, 3, x, L2, bn, activation, pool)
	x = residual_nopool_changeChannelnum(64, 3, x, L2, bn, activation, pool)
	x = residual_pool(64, 3, x, L2, bn, activation, pool)

	x = GlobalAveragePooling2D()(x)
	prediction = Dense(3, init='glorot_normal', activation='softmax', W_regularizer=l2(L2))(x)

	# compile the model 
	model = Model(input=inp, output= prediction)
	# sgd = SGD(lr=lr, decay=lr/n_epoch, momentum=0.9, nesterov=True)
	model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
	return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=13)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=13)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=13)
    plt.yticks(tick_marks, classes, size=13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=14)

    plt.tight_layout()
    plt.ylabel('True label', size=13)
    plt.xlabel('Predicted label', size=13)


print '-------------------------'
# load data
data = h5py.File('/raid/zyz293/NIST_data/NIST_newdata_split.hdf5')
test_file = np.array(data['test_file'])
test_label = np.array(data['test_label'])
del data

filepath = './weights.hdf5'
model = build_model()
model.load_weights(filepath)
test_batch = 3 * crop_size
test_step = len(test_file) // test_batch
test_generator = generator(test_file, test_label, False, test_batch)

pred_y = np.array(model.predict_generator(generator=test_generator, steps=test_step))
pred_y = np.array([sum(pred_y[i:i+crop_size]) for i in range(0, len(pred_y), crop_size)])
pred_y = np.argmax(pred_y, axis=1)
print pred_y.shape

true_label = np.array([test_label[i] for i in range(0, len(test_label), crop_size)])
print true_label.shape
acc = accuracy_score(np.array(true_label), pred_y)

sess = tf.Session()
print '------------------------'
print 'testing accuracy: ', acc