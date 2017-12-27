from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,source_url="http://yann.lecun.com/exdb/mnist/")

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape,  mnist.test.labels.shape)
print(mnist.validation.images.shape,   mnist.validation.images.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
# 输入为n*784
x = tf.placeholder(tf.float32, [None, 784])

# 定义参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 定义网络
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 优化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化
tf.global_variables_initializer().run()
# 每100为一个batch训练
for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

# 评估函数
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))