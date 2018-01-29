#here we go
import tensorflow as tf
import sqlite3
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

nodes1 = 500
nodes2 = 500
nodes3 = 500

classes = 10
batch_size = 50

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

def neural_network(data):
    # (input_data * weights) + biases

    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, nodes1])),\
                     'biases':tf.Variable(tf.random_normal([nodes1]))}
    
    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([nodes1, nodes2])),\
                     'biases':tf.Variable(tf.random_normal([nodes2]))}
    
    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([nodes2, nodes3])),\
                     'biases':tf.Variable(tf.random_normal([nodes3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes3, classes])),\
                     'biases':tf.Variable(tf.random_normal([classes]))}

    L1 = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['biases'])
    L1 = tf.nn.relu(L1)#for sigmoid function, basically an activation function

    L2 = tf.add(tf.matmul(L1,hidden_layer2['weights']), hidden_layer2['biases'])
    L2 = tf.nn.relu(L2)

    L3 = tf.add(tf.matmul(L2,hidden_layer3['weights']), hidden_layer3['biases'])
    L3 = tf.nn.relu(L3)

    output = tf.add(tf.matmul(L3,output_layer['weights']), output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #epochs = cyles of feed forward + backprop
    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x, e_y = mnist.train.next_batch(batch_size)#make next batch
                _, c = sess.run([optimizer, cost], feed_dict={x: e_x, y: e_y})
                epoch_loss += c
            print("Epoch", epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)

