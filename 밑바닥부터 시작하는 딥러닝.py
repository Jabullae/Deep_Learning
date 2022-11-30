# 함수선언
def AND(x1, x2):
    w1, w2, theta = 0.4, 0.4, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def NAND(x1, x2):
    w1, w2, theta = -0.4, -0.4, -0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 0.4, 0.4, 0.3
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def XOR(x1, x2):
    y1 = NAND(x1, x2)
    y2 = OR(x1, x2)
    return AND(y1, y2)

# 함수적용
print(NAND(0, 0), 'AND', OR(0, 0), '=>', XOR(0, 0))
print(NAND(1, 0), 'AND', OR(1, 0), '=>', XOR(1, 0))
print(NAND(0, 1), 'AND', OR(0, 1), '=>', XOR(0, 1))
print(NAND(1, 1), 'AND', OR(1, 1), '=>', XOR(1, 1))
print(NAND(1, 1), 'AND', OR(1, 1), '=>', XOR(1, 1))

######################################################

import numpy as np
import matplotlib.pylab as plt


x = np.arange(-6, 6, 0.01)


#######################################################
# 선형함수들
#######################################################
def identity_func(x): # 항등함수
    return x

#그래프 출력
plt.plot(x, identity_func(x), linestyle='--', label="identity")


def linear_func(x): # 1차함수
    return 1.5 * x + 1 # a기울기(1.5), Y절편b(1) 조정가능

#그래프 출력
plt.plot(x, linear_func(x), linestyle='--', label="linear")
plt.legend()
plt.show()


#######################################################
# 계단함수들
#######################################################
def binarystep_func(x): # 계단함수
    return (x>=0)*1
    # return np.array(x>=0, dtype = np.int) # same result

    # y = x >= 0
    # return y.astype(np.int) # Copy of the array, cast to a specified type.
    # same result

#그래프 출력
plt.plot(x, binarystep_func(x), linestyle='--', label="binary step")


def sgn_func(x): # 부호함수(sign function)
    return (x>=0)*1 + (x<=0)*-1

#그래프 출력
plt.plot(x, sgn_func(x), linestyle='--', label="sign function")
plt.legend()
plt.show()


#######################################################
# Sigmoid계열
#######################################################

def softstep_func(x): # Soft step (= Logistic), 시그모이드(Sigmoid, S자모양) 대표적인 함수
    return 1 / (1 + np.exp(-x))

#그래프 출력
plt.plot(x, softstep_func(x), linestyle='--', label="Soft step (= Logistic)")

def tanh_func(x): # TanH 함수
    return np.tanh(x)
    # return 2 / (1 + np.exp(-2*x)) - 1 # same

#그래프 출력
plt.plot(x, tanh_func(x), linestyle='--', label="TanH")


def arctan_func(x): # ArcTan 함수
    return np.arctan(x)

#그래프 출력
plt.plot(x, arctan_func(x), linestyle='--', label="ArcTan")


def softsign_func(x): # Softsign 함수
    return x / ( 1+ np.abs(x) )

#그래프 출력
plt.plot(x, softsign_func(x), linestyle='--', label="Softsign")
plt.legend()
plt.show()



#######################################################
# ReLU계열
#######################################################

def relu_func(x): # ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>0)*x
    # return np.maximum(0,x) # same

#그래프 출력
plt.plot(x, relu_func(x), linestyle='--', label="ReLU")


def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>=0)*x + (x<0)*0.01*x # 알파값(보통 0.01) 조정가능
    # return np.maximum(0.01*x,x) # same

#그래프 출력
plt.plot(x, leakyrelu_func(x), linestyle='--', label="Leaky ReLU")


def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)

#그래프 출력
plt.plot(x, elu_func(x), linestyle='--', label="ELU(Exponential linear unit)")


def trelu_func(x): # Thresholded ReLU
    return (x>1)*x # 임계값(1) 조정 가능

#그래프 출력
plt.plot(x, trelu_func(x), linestyle='--', label="Thresholded ReLU")
plt.legend()
plt.show()



#######################################################
# 기타계열
#######################################################

def softplus_func(x): # SoftPlus 함수
    return np.log( 1 + np.exp(x) )

#그래프 출력
plt.plot(x, softplus_func(x), linestyle='--', label="SoftPlus")


def bentidentity_func(x): # Bent identity
    return (np.sqrt(x*x+1)-1)/2+x

#그래프 출력
plt.plot(x, bentidentity_func(x), linestyle='--', label="Bent identity")


def gaussian_func(x): # Gaussian
    return np.exp(-x*x)

#그래프 출력
plt.plot(x, gaussian_func(x), linestyle='--', label="Gaussian")

#plt.plot(x, y_identity, 'r--', x, relu_func(x), 'b--', x, softstep_func(x), 'g--')
plt.ylim(-5, 5)
plt.legend()
plt.show()

###########################################################################
# 1차원 배열
import numpy as numpy
A = np.array([1, 2, 3, 4])

np.ndim(A)        # 배열의 차원 수 확인, 1차원!
A.shape            # (4, )
A.shape[0]        # 4

# 2차원 배열
import numpy as numpy
B = np.array([[1, 2], [3, 4], [5, 6]])
np.ndim(B)        # 2 -> 2차원이다!
B.shape            # (3, 2)

# 2차원 배열의 내적
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A,B))        # [[19 22]
                         #  [43 50]]

# 퍼셉트론의 관점

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)

print(Y)
