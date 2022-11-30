def GATE(x1, x2):
    w1, w2, theta = -0.4, -0.4, -0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

GATE(0, 0)
GATE(0, 1)
GATE(1, 0)
GATE(1, 1)


1+1
