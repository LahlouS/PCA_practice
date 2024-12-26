def f(x):
    if x < -1:
        return -x
    elif x < 1:
        return 2*x+3
    else:
        return 3-2*x

def g(u):
    return u*u-u-1

for k in range(11):
    print(k-5, g(f(k)), f(g(k)))

    