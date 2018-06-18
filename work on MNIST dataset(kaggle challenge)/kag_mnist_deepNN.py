import tensorflow as tf
import pandas as pd
import numpy as np
import random as rn


seed = 128
rng = np.random.RandomState(seed)

train = pd.read_csv("..../train.csv")
test = pd.read_csv(".../test.csv")

n=40000
#Xtrain =df[0:n,1:]
train_x = train.iloc[0:n,1:].values
#train_x = train_x/255
Ytrain =train.iloc[0:n,0].values
test_x = train.iloc[n:,1:].values
Ytest =train.iloc[n:,0].values
#test_label = df[n:,0]
Xtest=test.iloc[0:28000].values

nodes_hl1 = 500
nodes_hl2 = 600
nodes_hl3 = 700

classes = 10
batch_size = 100
input_num_units = 784
learning_rate = 0.005

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([784,nodes_hl1])), 'biases': tf.Variable(tf.random_normal([nodes_hl1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])), 'biases': tf.Variable(tf.random_normal([nodes_hl2]))}

    hl3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])), 'biases': tf.Variable(tf.random_normal([nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl3, classes])), 'biases': tf.Variable(tf.random_normal([classes]))}

    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)

    output =tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output



def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y
def train_neural_network(x):
            prediction = neural_network_model(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #learning rate by default = 0.001
            print("learning rate = " + str(learning_rate) +" :: batch size = "+str(batch_size))
            epochs = 10
            with tf.Session() as sess:
                   sess.run(tf.global_variables_initializer())

                   for epoch in range(epochs):
                        loss= 0
                        total_batch = int(train_x.shape[0]/batch_size)
                        for i in range(total_batch):
                            epoch_x ,epoch_y = batch_creator(batch_size, train_x.shape[0], 'train')

                            #epoch_x,epoch_y = mnist.train.next_batch(batch_sizes)
                            _, c = sess.run([optimizer,cost], feed_dict = {x:epoch_x, y:epoch_y})
                            loss +=c
                        print("Epoch " + str(epoch+1) + " completed out of " + str(epochs) + " ::: loss -> "+str(loss))

                   correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                   accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                  
                   print("Accuracy : "+ str(accuracy.eval({x: test_x.reshape(-1, input_num_units), y: dense_to_one_hot(Ytest)})))

             
                   predict = tf.argmax(prediction, 1)
                   pred = predict.eval({x: Xtest.reshape(-1, input_num_units)})

                   df = pd.DataFrame({'Label': pred})
                    # Add 'ImageId' column
                   df1 = pd.concat([pd.Series(range(1,28001), name='ImageId'), df[['Label']]], axis=1)
                   df1.to_csv('submission_mnist.csv', index=False)

train_neural_network(x)
