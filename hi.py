# 함수선언
def AND(x1, x2):
    w1, w2, theta = 0.4, 0.4, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def NAND(x1, x2):
    w1, w2, theta = -0.4, -0.4, -0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 0.4, 0.4, 0.3
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def XOR(x1, x2):
    y1 = NAND(x1, x2)
    y2 = OR(x1, x2)
    return AND(y1, y2)

# 함수적용
print(NAND(0, 0), 'AND', OR(0, 0), '=>', XOR(0, 0))
print(NAND(1, 0), 'AND', OR(1, 0), '=>', XOR(1, 0))
print(NAND(0, 1), 'AND', OR(0, 1), '=>', XOR(0, 1))
print(NAND(1, 1), 'AND', OR(1, 1), '=>', XOR(1, 1))
print(NAND(1, 1), 'AND', OR(1, 1), '=>', XOR(1, 1))
