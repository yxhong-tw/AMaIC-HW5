import numpy as np

from typing import Callable, List, Tuple

x = np.arange(start=0, stop=5.05, step=0.1)
n = len(x)
y = 1 + (3 * x) + (2 * (np.random.rand(n) - np.random.rand(n)))


def criteria(a: np.ndarray) -> np.float64:
    total_loss = 0

    for i in range(n):
        # y = a_{0} + (a_{1} * x)
        estimation = a[0] + (a[1] * x[i])
        total_loss += (estimation - y[i])**2

    return (total_loss / n)


def get_gradients(
    criteria: Callable,
    _x: List[int],
    h: float = 1e-5,
) -> np.ndarray[np.float64]:
    _n = len(_x)
    dm = h * np.eye(_n)
    gradients = []

    # Central Difference
    for i in range(_n):
        gradients.append(
            (criteria(a=(_x + dm[:, i])) - criteria(a=(_x - dm[:, i]))) /
            (2 * h))

    return np.array(gradients)


def get_norm(gradients: np.ndarray[np.float64]) -> np.float64:
    norm = 0

    for i in range(len(gradients)):
        norm += gradients[i]**2

    return np.sqrt(norm)


def bracket(f: Callable, x1: float, h: float) -> Tuple[float, float]:
    c = 1.618033989

    f1 = f(point=x1)
    x2 = x1 + h
    f2 = f(point=x2)

    if f2 > f1:
        h *= -1
        x2 = x1 + h
        f2 = f(point=x2)

        if f2 > f1:
            return x2, (x1 - h)

    for _ in range(100):
        h = c * h
        x3 = x2 + h
        f3 = f(point=x3)

        if f3 > f2:
            return x1, x3

        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3

    print("The bracket does not include a minimum.")


def binary_search(
    f: Callable,
    interval: List[float],
    tolerance: float = 1.0e-6,
    max_iter: int = 100,
) -> Tuple[float, np.float64, int]:
    a = interval[0]
    b = interval[1]
    delta = (tolerance / 2)
    current_iter = 0

    while (b - a) > tolerance and current_iter < max_iter:
        current_iter += 1

        mid = (a + b) / 2
        x1 = mid - delta
        f_x1 = f(point=x1)
        x2 = mid + delta
        f_x2 = f(point=x2)

        if f_x1 < f_x2:
            b = x2
        else:
            a = x1

    opt = (a + b) / 2

    return opt, f(point=opt), (current_iter + 1)


def steepest_descent(
    criteria: Callable,
    _x: List[float],
    h: float = 1e-5,
    max_iter: int = 1000,
) -> Tuple[np.ndarray[np.float64], int]:
    for current_iter in range(max_iter):
        if current_iter == (max_iter - 1):
            print("Maximum number of iteration exceeded!")

        gradients = -get_gradients(criteria=criteria, _x=_x)

        print(f"Current Iteration: {current_iter}; Gradients: {gradients}")

        if get_norm(gradients=gradients) < h:
            break

        def get_loss(point: float) -> np.float64:
            return criteria(a=(_x + (point * gradients)))

        a, b = bracket(f=get_loss, x1=0.0, h=0.1)
        opt, _, _ = binary_search(f=get_loss, interval=[a, b])
        _x = _x + (opt * gradients)

    return _x, (current_iter + 1)


if __name__ == "__main__":
    guess = [2, 2]
    opt, iter = steepest_descent(criteria=criteria, _x=guess)

    print(
        f"Using Steepest Descent (with optimal learning rate from 1D optimization) after {iter} iterations, the coefs are: {opt}."
    )
