from sympy import var
from sympy import solve

l1y1, l1y2, l2y1, l2y2, l1x1, l1x2, l2x1, l2x2, D2, E1, E2, xc, yc, k, Ys = var(
    "l1y1 l1y2 l2y1 l2y2 l1x1 l1x2 l2x1 l2x2 D2 E1 E2 xc yc k Ys"
)

print(solve(
    [
        (l1x1 - xc) ** 2 - E1 * Ys * (l1y1 - yc) ** 2 - E1*D2,
        (l1x2 - xc) ** 2 - E1 * Ys * (l1y2 - yc) ** 2 - E1*D2,
        (l2x1 - xc) ** 2 - E2 * Ys * (l2y1 - yc) ** 2 - E2*D2,
        (l2x2 - xc) ** 2 - E2 * Ys * (l2y2 - yc) ** 2 - E2*D2,
    ],
    [xc, yc, D2, Ys],
    dict=True,
))