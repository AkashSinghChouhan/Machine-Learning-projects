import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing,model_selection
import seaborn as sns
from  sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import tensorflow as tf
#Variable	Definition	Key
#survival	Survival	0 = No, 1 = Yes
#pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
#sex	Sex	
#Age	Age in years	
#sibsp	# of siblings / spouses aboard the Titanic	
#parch	# of parents / children aboard the Titanic	
#ticket	Ticket number	
#fare	Passenger fare	
#cabin	Cabin number	
#embarked	Port of Embarkation

train =pd.read_csv("C:/Users/NEO/Downloads/titanic dataset/train.csv")

train.drop(['Name'], 1 , inplace=True)
train.convert_objects(convert_numeric=True)

# absolute numbers
print(train["Survived"].value_counts())

# percentages
print(train["Survived"].value_counts(normalize = True))

#print(df["Survived"][df["Sex"] == 'male'].value_counts())
#print(df["Survived"][df["Sex"] == 'female'].value_counts())

pd.options.mode.chained_assignment = None 
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1



#FILLING THE MISSING DATA

train["Age"] =train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("C")
train["Embarked"][train["Embarked"] == 'S'] = 0
train["Embarked"][train["Embarked"] == 'C'] = 1
train["Embarked"][train["Embarked"] == 'Q'] = 2

#function courtsey to sentdex
def handle_non_numerical_data(train):
    columns = train.columns.values

    for column in columns:
        text_digit_vals = { }
        def convert_to_int(val):
            return text_digit_vals[val]

        if train[column].dtype != np.int64 and train[column].dtype != np.float64:
            column_contents = train[column].values.tolist()
            unique_elements = set(column_contents)
            x= 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            train[column] = list(map(convert_to_int, train[column]))
    return train

train = handle_non_numerical_data(train)

data_col= ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Cabin','Ticket','Fare','Embarked']
P=train[data_col].values
q=train["Survived"].values

seed = 10
rng = np.random.RandomState(seed)

train_x,X_test,y_train,y_test= train_test_split(P,q,random_state= 4)

nodes_hl1 = 100
nodes_hl2 = 100
nodes_hl3 = 100
nodes_hl4 = 100
nodes_hl5 = 100
nodes_hl6 = 100
nodes_hl7 = 100
nodes_hl8 = 100
nodes_hl9 = 100
nodes_hl10 = 100

classes = 2
batch_size = 100
input_num_units = 10
learning_rate = 0.001

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([10,nodes_hl1])), 'biases': tf.Variable(tf.random_normal([nodes_hl1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])), 'biases': tf.Variable(tf.random_normal([nodes_hl2]))}

    hl3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])), 'biases': tf.Variable(tf.random_normal([nodes_hl3]))}

    hl4 = {'weights': tf.Variable(tf.random_normal([nodes_hl3, nodes_hl4])), 'biases': tf.Variable(tf.random_normal([nodes_hl4]))}
    hl5 = {'weights': tf.Variable(tf.random_normal([nodes_hl4, nodes_hl5])), 'biases': tf.Variable(tf.random_normal([nodes_hl5]))}
    hl6 = {'weights': tf.Variable(tf.random_normal([nodes_hl5, nodes_hl6])), 'biases': tf.Variable(tf.random_normal([nodes_hl6]))}
    hl7 = {'weights': tf.Variable(tf.random_normal([nodes_hl6, nodes_hl7])), 'biases': tf.Variable(tf.random_normal([nodes_hl7]))}
    hl8 = {'weights': tf.Variable(tf.random_normal([nodes_hl7, nodes_hl8])), 'biases': tf.Variable(tf.random_normal([nodes_hl8]))}
    hl9 = {'weights': tf.Variable(tf.random_normal([nodes_hl8, nodes_hl9])), 'biases': tf.Variable(tf.random_normal([nodes_hl9]))}
    hl10 = {'weights': tf.Variable(tf.random_normal([nodes_hl9, nodes_hl10])), 'biases': tf.Variable(tf.random_normal([nodes_hl10]))}

     

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl10, classes])), 'biases': tf.Variable(tf.random_normal([classes]))}

    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.sigmoid(l3)

    l4 = tf.add(tf.matmul(l3 ,hl4['weights']), hl4['biases'])
    l4 = tf.nn.sigmoid(l4)

    l5 = tf.add(tf.matmul(l4, hl5['weights']), hl5['biases'])
    l5 = tf.nn.sigmoid(l5)

    l6 = tf.add(tf.matmul(l5, hl6['weights']), hl6['biases'])
    l6 = tf.nn.sigmoid(l6)

    l7 = tf.add(tf.matmul(l6, hl7['weights']), hl7['biases'])
    l7 = tf.nn.sigmoid(l7)

    l8 = tf.add(tf.matmul(l7, hl8['weights']), hl8['biases'])
    l8 = tf.nn.sigmoid(l8)

    l9 = tf.add(tf.matmul(l8, hl9['weights']), hl9['biases'])
    l9 = tf.nn.sigmoid(l9)

    l10 = tf.add(tf.matmul(l9, hl10['weights']), hl10['biases'])
    l10 = tf.nn.sigmoid(l10)


    output =tf.add(tf.matmul(l10, output_layer['weights']), output_layer['biases'])

    return output


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'Survived'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

def train_neural_network(x):
            prediction = neural_network_model(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) #learning rate by default = 0.001
            print("learning rate = " + str(learning_rate) +" :: batch size = "+str(batch_size))
            epochs = 30
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
                  
                   print("Accuracy : "+ str(accuracy.eval({x: X_test.reshape(-1, input_num_units), y: dense_to_one_hot(y_test)})))

             
                   """predict = tf.argmax(prediction, 1)
                   pred = predict.eval({x: Xtest.reshape(-1, input_num_units)})

                   df = pd.DataFrame({'Label': pred})
                    # Add 'ImageId' column
                   df1 = pd.concat([pd.Series(range(1,28001), name='ImageId'), df[['Label']]], axis=1)
                   df1.to_csv('submission.csv', index=False)"""

train_neural_network(x)


