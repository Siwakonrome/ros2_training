import numpy as np
import matplotlib.pyplot as plt

list_1 = [1,2,3,4]
list_2 = [2,3,4,5]

vec_1 = np.array(list_1)
vec_2 = np.array(list_2)

"""
print(list_1 + list_2)
print(vec_1 * vec_2)
print(np.dot(vec_1,vec_2))

Result
[1, 2, 3, 4, 2, 3, 4, 5]
[ 2  6 12 20]
40
"""


a = np.array([1,2,0.5])
b = np.arange(0,10)
c = np.arange(0,10,2)
d = np.linspace(0,10,11)

"""
print(a)
print(b)
print(c)
print(d)

Result
[1.  2.  0.5]
[0 1 2 3 4 5 6 7 8 9]
[0 2 4 6 8]
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
"""



N = 100
t = np.linspace(0, 10, N+1)
y = np.sin(t)

"""
plt.plot(t,y)
plt.legend("y(t)")
plt.ylabel("distance [m]")
plt.xlabel("time [s]")
plt.show()

Result
print(np.linalg.norm(a))
print(a/np.linalg.norm(a))
2.29128784747792
[0.43643578 0.87287156 0.21821789]
"""

a_radom = np.random.rand((12))
A = np.array([[1,2,3,4,5],[5,6,7,8,9]])

"""
print(a_radom)
print(A.shape)
print(A)
print(A[0,1:-1])

Result
[0.90097719 0.4495457  0.60000931 0.63163379 0.03483784 0.08190502
 0.56331951 0.36540942 0.19472268 0.41666803 0.41088183 0.76143552]
(2, 5)
[[1 2 3 4 5]
 [5 6 7 8 9]]
[2 3 4]
"""

B = np.array([[5,6,7,8,9]]).T

"""
print(np.matmul(A,B))
[[115]
 [255]]
"""

A_rad = np.random.randint(0,10,size=(5,6))
B_rad = np.random.normal(1,0.2,(5,6))

"""
print(A_rad)
print(B_rad)

Result
[[4 0 5 9 4 3]
 [3 7 6 0 4 8]
 [8 9 2 0 3 4]
 [2 2 4 9 9 3]
 [3 6 9 3 4 4]]
[[1.15534119 0.80264615 0.6894397  0.91340257 0.96647485 0.77383934]
 [0.58152623 0.75857929 1.05588685 0.85455865 1.07462709 1.50388884]
 [0.84561504 1.20275799 1.22881171 1.16036659 0.61647151 0.94705942]
 [0.85260014 0.75494064 1.21839302 1.18796065 0.98915755 0.87441161]
 [0.90416438 0.61408434 0.84428543 1.08509137 1.04859763 1.06027937]]
"""



A_1 = np.array([[1,2,3]])
B_1 = np.array([[4,5,6],[7,8,9]])

"""
print(np.concatenate((A_1,B_1)))

Result
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

X = np.array([[0,0,1]]).T
Y = np.array([[0,-1,0]]).T
Z = np.array([[1,0,0]]).T
Result = np.concatenate((X,Y,Z), axis=1)

"""
print(Result)

Result
[[ 0  0  1]
 [ 0 -1  0]
 [ 1  0  0]]
"""






