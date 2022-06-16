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
"""

print(np.linalg.norm(a))
print(a/np.linalg.norm(a))

a_radom = np.random.rand((12))
A = np.array([[1,2,3,4,5],[5,6,7,8,9]])
"""
print(a_radom)
print(A.shape)
print(A)
print(A[0,1:-1])
"""
B = np.array([[5,6,7,8,9]]).T
#print(np.matmul(A,B))

A_rad = np.random.randint(0,10,size=(5,6))
B_rad = np.random.normal(1,0.2,(5,6))
"""
print(A_rad)
print(B_rad)
"""

A_1 = np.array([[1,2,3]])
B_1 = np.array([[4,5,6],[7,8,9]])

#print(np.concatenate((A_1,B_1)))

X = np.array([[0,0,1]]).T
Y = np.array([[0,-1,0]]).T
Z = np.array([[1,0,0]]).T
Result = np.concatenate((X,Y,Z), axis=1)

"""
print(Result)
"""





