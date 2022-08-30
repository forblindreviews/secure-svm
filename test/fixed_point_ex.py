import math
N = 9
a = 1.1231432412
print(a)
a_fxp = int(a * math.pow(2, N))
print(a_fxp)

a_float = a_fxp * 1.0 / math.pow(2, N) 
print(a_float)