import numpy as np
a = np.array([[1, 2, 3], [4, 4, 1]])
print(a)
a = np.array([map(lambda x:x if x > 3 else 100, a)])
#map(lambda x:x if x!= 4 else 'sss',a)
print(a)
