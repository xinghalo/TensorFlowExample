import tensorflow as tf

# argmax，取tensor里面最大的那个数
argmax_r = tf.argmax(tf.constant([[0.1,0.9,0.3],[0.0,0.2,0.8]]), 1)
print(argmax_r)

# equal

equal_r = tf.equal(tf.constant([1, 2, 3, 4],1),tf.constant([1,1,1,1],1))
print(equal_r)

