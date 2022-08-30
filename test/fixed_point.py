from fxpmath import Fxp
import numpy as np

x = np.array([[1/3, 1/3]])
y = np.array([[1/3], [1/3]])

print("X original = ", x)
print("Y original = ", y)

print("-----------")

x_fxp = Fxp(x, signed=False, n_word=20, n_frac=15)
y_fxp= Fxp(y, signed=False, n_word=20, n_frac=15)

print("X fxp =", x_fxp.get_val())
print("Y fxp =", y_fxp.get_val())

print("-----------")

print(x_fxp.info(verbose=3))

print("-----------")
print("Dot product without scaling = ", x_fxp.get_val().dot(y_fxp.get_val()))
print("Dot prod scaled = ", Fxp(x_fxp.get_val().dot(y_fxp.get_val()), n_word=20, n_frac=15))
print("Dot prod without specifications = ", Fxp(x_fxp.get_val().dot(y_fxp.get_val())))


dot_fpx = Fxp(None, signed=True, n_word=20, n_frac=15)
dot_fpx.equal(x_fxp().dot(y_fxp()))

print("Dot with new variable = ", dot_fpx())