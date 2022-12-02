! pip install tensorflow

import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
grad_of_y_wrt_x

x = tf.Variable(tf.zeros((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
grad_of_y_wrt_x

W  = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2, )))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])
grad_of_y_wrt_W_and_b

# 상수 텐서와 변수
## 모두 1또는 0인 텐서
import tensorflow as tf

x = tf.ones(shape=(2, 1))
print(x)

x = tf.zeros(shape = (2, 1))
print(x)

## 랜덤 텐서
x = tf.random.normal(shape = (3, 1), mean = 0, stddev=1.)
print(x)

x = tf.random.uniform(shape = (3, 1), minval=0, maxval=1.)
print(x)

## 넘파이 배열값에 값 할당하기
