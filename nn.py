import numpy as np
mat=np.array([[0,0.5,0.5,0,0],[1/3,0,1/3,1/3,0],[1/3,1/3,0,1/3,0],[0,1/3,1/3,0,1/3],[0,0,0,1,0]])
x=[1/6,1/4,1/4,1/4,1/12]
b=np.matmul(x,mat)
print(b)
import math
x=(280-3.5*70)/math.sqrt(70*35/12)
print(x)
# import numpy as np
# # A=np.array([[4,2,2],[2,4,2],[2,2,4]])
# # a,b=np.linalg.eig(A)
# # print(a)
# # print(math.sqrt())
print(35/math.sqrt(1225/6))