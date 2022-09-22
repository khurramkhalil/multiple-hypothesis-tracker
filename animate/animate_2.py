import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i] * x ** i - (x ** i / 100)
    return y


# x = np.linspace(0, 9, 20)
# polynomial = [1, 0.2, 3, 0.4, 5]
# y = PolyCoefficients(x, polynomial)

polynomial = [1, 2, 3, 4, ]
lower_limit = -40
upper_limit = 40
num_pts = 150
x = np.linspace(lower_limit, upper_limit, num_pts)

poly_coefs = polynomial[::-1]  # [4, 3, 2, 1]
y = np.polyval(poly_coefs, x) / 3000

x1 = np.linspace(upper_limit, lower_limit, num_pts)
y1 = np.polyval(poly_coefs, x) / 3000 - 10

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers'))

# plt.plot(x, PolyCoefficients(x, coeffs))
fig.show()

# import numpy as np
# from matplotlib import pyplot as plt
#
# x = np.linspace(0, 10, 11)
# coeffs = [1, 2, 3, 4, 5]
# y = np.array([np.sum(np.array([coeffs[i]*(j**i) for i in range(len(coeffs))])) for j in x])
# plt.plot(x, y)
# plt.show()

# polynomial = [1, 2, 3, 4]
# lower_limit = -2
# upper_limit = 2
# num_pts = 100
# x = np.linspace(lower_limit, upper_limit, num_pts)
#
# poly_coefs = polynomial[::-1]  # [4, 3, 2, 1]
# y = np.polyval(poly_coefs, x)
# plt.plot(x, y, '-r')
