inputs = [1.0, -2.0, 3.0]
weights = [-3.0, -1.0, 2.0]
bias = 1.0;

#first input inputs[0] and weights weights[0]

iw0 = inputs[0] * weights[0]
iw1 = inputs[1] * weights[1]
iw2 = inputs[2] * weights[2]

print(iw0,iw1,iw2)
z = iw0 + iw1 + iw2 + bias
print(z)

## ReLU

y = max(z,0)
print(y)