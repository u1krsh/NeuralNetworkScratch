import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x =np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

##close enough point
p2_delta = 0.0001
x1 = 200
x2 = x1+p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1,y1), (x2, y2))

approx_derivative = (y2-y1)/(x2-x1)
#y intercept
b = y2 - approx_derivative*x2


def tangent_line(x):
    return approx_derivative*x+b

to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot, [tangent_line(i) for i in to_plot])

print('Approx Derivative', f'where x = {x1} is {approx_derivative}')
plt.show()