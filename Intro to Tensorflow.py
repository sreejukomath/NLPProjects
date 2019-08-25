#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:02:13 2017

@author: Kanth
"""

import tensorflow as tf
import numpy as np

#Rank
scalar = tf.constant(100)
vector = tf.constant([1,2,3,4,5])
matrix = tf.constant([[1,2,3],[4,5,6]])
cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
print(scalar.get_shape())
print(vector.get_shape())
print(matrix.get_shape())
print(cube_matrix.get_shape())

#Shape:

scalar.get_shape()

#Operators

tensor_1 = np.array([1,2,3,4,5,6,7,8,9,10])
tensor_1 = tf.constant(tensor_1)
with tf.Session() as sess:

    print(tensor_1.get_shape())
    print(sess.run(tensor_1))
    
tensor_2 = np.array([(1,2,3),(4,5,6),(7,8,9)])
tensor_2 = tf.Variable(tensor_2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tensor_2.get_shape())
    print(tensor_2.get_shape())
    print(sess.run(tensor_2))
    
#Variables
value = tf.Variable(0, name="value")   
one = tf.constant(1) 
new_value = tf.add(value,one)
update_value= tf.assign(value,new_value)
initialize = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initialize)
    print(sess.run(value))
    for i in range (10):
        sess.run(update_value)
        print(sess.run(value))
        
#Fetchccsvccc
constant_a = tf.constant([300.0])
constant_b = tf.constant([100.0])
constant_c = tf.constant([3.0])
sum = tf.add(constant_a, constant_b)
mul = tf.multiply(constant_a,constant_c)

with tf.Session() as sess:
    result = sess.run([sum,mul])
    print(result)
    
#Feed

a=2
b=3
x = tf.placeholder(tf.float32, shape=(a,b))
y = tf.add(a,b)
data = np.random.rand(a,b)
sess = tf.Session()
print(sess.run(y, feed_dict={x:data}))