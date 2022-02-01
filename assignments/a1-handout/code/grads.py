from turtle import shape
import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 2
    λ = 6  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result


def foo_grad(x):
    res = np.array([])
    for x_i in x:
        res = np.append(res, [6*x_i**5], axis=0)
    return res

def bar(x):
    return np.prod(x)


def bar_grad(x):
    # Implementation needs to be here.
    res = np.empty(shape=(len(x)))
    for i in range(len(x)):
        res[i] = np.prod(x[:i]) * np.prod(x[i+1:])
    return res

# Hint: This is a bit tricky - what if one of the x[i] is zero?
