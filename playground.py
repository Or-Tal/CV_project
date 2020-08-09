from tensorflow import keras as K
import tensorflow as tf


if __name__ == '__main__':
    A = tf.constant(value=[[[1, 1],[1,1]],[[2, 2],[2,2]]])
    B = tf.concat([A, A], -1)
    print(A)
    print(B)