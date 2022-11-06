def zero_crossing(x: list, y: list) -> float:
    m = (y[1]-y[0])/(x[1]-x[0])
    c = y[0] - m * x[0]
    return -c/m

def lin_y(x_eval, x: list, y: list) -> float:
    m = (y[1]-y[0])/(x[1]-x[0])
    c = y[0] - m * x[0]
    y_eval = x_eval * m + c
    return y_eval