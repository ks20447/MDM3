import simpy as sp
import numpy as np
import random as rn


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    env = sp.Environment()
    x = rn.randint(1, 10)
    print(x)
    y = np.array([1, 2, 3])
    print(y)
    print("Hello World")


if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

