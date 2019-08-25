# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:04:46 2019

@author: Sreeju
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
           cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

session = tf.Session()

num_classes = 10

# validation split
validation_size = 0.2
# batch size
batch_size = 16
data = train_data
    
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
    
def create_convolutional_layer(input,
              num_input_channels,
              conv_filter_size,
              num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input,
                    filter=weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
   layer_shape = layer.get_shape()
   num_features = layer_shape[1:4].num_elements()
   layer = tf.reshape(layer, [-1, num_features])
   return layer    

def create_fc_layer(input,
            num_inputs,
            num_outputs,
            use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
       layer = tf.nn.relu(layer)
    return layer

image_width = train_data[0].shape[0]
image_height = train_data[0].shape[1]

num_channels = 3
filter_size_conv1 = 2
num_filters_conv1 = 1
fc_layer_size = 2
learning_rate = 0.005

x = tf.placeholder(tf.float32, shape=[batch_size, image_width,image_height,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1 = create_convolutional_layer(input=x,
              num_input_channels=num_channels,
              conv_filter_size=filter_size_conv1,
              num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
              num_input_channels=num_filters_conv1,
              conv_filter_size=filter_size_conv1,
              num_filters=num_filters_conv1)
layer_conv3= create_convolutional_layer(input=layer_conv2,
              num_input_channels=num_filters_conv1,
              conv_filter_size=filter_size_conv1,
              num_filters=num_filters_conv1)
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(input=layer_flat,
                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                    num_outputs=fc_layer_size,
                    use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1,
                    num_inputs=fc_layer_size,
                    num_outputs=num_classes,
                    use_relu=False)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                   labels=y_true)
cost = tf.reduce_mean(cross_entropy)

rand_index = np.random.choice(len(data), size=batch_size)
x_batch = data[rand_index]
x_batch = np.expand_dims(x_batch, 3)
y_true_batch = train_labels[rand_index]

print (x_batch.shape)
    
feed_dict_train = {x: x_batch, y_true: y_true_batch}
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
session.run(optimizer, feed_dict=feed_dict_train)

train_loss = []

evaluation_size = 500

def trainModel(num_iteration):
   global total_iterations
   total_iterations = 0
   for i in range(total_iterations, total_iterations + num_iteration):
       rand_index = np.random.choice(len(data), size=batch_size)
       x_batch = data[rand_index]
       x_batch = np.expand_dims(x_batch, 3)
       y_true_batch = train_labels[rand_index]

       eval_index = np.random.choice(len(test_data), size=evaluation_size)
       x_valid_batch = test_data[eval_index]
       x_valid_batch = np.expand_dims(x_valid_batch, 3)
       y_valid_batch = test_labels[eval_index]

       feed_dict_tr = {x: x_batch, y_true: y_true_batch}
       feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

       session.run(optimizer, feed_dict=feed_dict_tr)
       if i % int(data.train.num_examples/batch_size) == 0:
           val_loss = session.run(cost, feed_dict=feed_dict_val)
           train_loss.append(val_loss)           
           print("loss: ", val_loss)
           epoch = int(i / int(data.train.num_examples/batch_size))
           print("epoch: ", epoch)
           #show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
           #saver.save(session, 'cfar10-model')

trainModel(10)   
# Matlotlib code to plot the loss
eval_indices = range(0, total_iterations, 5)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()