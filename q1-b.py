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


def gradient_descent(
    criteria: Callable,
    _x: List[float],
    h: float = 1e-4,
    max_iter: int = 5000,
    lr: float = 0.01,
    momentum: float = 0.9,
) -> Tuple[np.ndarray[np.float64], int]:
    _n = len(_x)
    gradients = get_gradients(criteria=criteria, _x=_x)
    velocities = np.zeros(_n)

    for current_iter in range(max_iter):
        if current_iter == (max_iter - 1):
            print("Maximum number of iteration exceeded!")

        for i in range(_n):
            velocities[i] = (momentum * velocities[i]) - (lr * gradients[i])
            _x[i] += velocities[i]

        gradients = get_gradients(criteria=criteria, _x=_x)

        print(f"Current Iteration: {current_iter}; Gradients: {gradients}")

        if get_norm(gradients=gradients) < h:
            break

    return _x, (current_iter + 1)


if __name__ == "__main__":
    guess = [2, 2]

    opt, iter = gradient_descent(criteria=criteria, _x=guess)
    print(
        f"Using Gradient Descent (with momentum) after {iter} iterations, the coefs are: {opt}."
    )
