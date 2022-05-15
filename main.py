"""
Camera pose estimation / Space resection

## Example:

`$ python ./main -i points.txt -f 153.24 -x 0 -y 0 -m 40000 -n 100`

where `points.txt' is formatted as:
```-86.15 -68.99 36589.41 25273.32 2195.17
-53.40 82.21 37631.08 31324.51 728.69
-14.78 -76.63 39100.97 24934.98 2386.50
10.46 64.43 40426.54 30319.81 757.31```

## Reference:
* 武汉大学 摄影测量学 主讲-袁修孝 视频教程
    * https://www.bilibili.com/video/BV19x411B7ZV?p=28
    * https://www.bilibili.com/video/BV19x411B7ZV?p=29
    * https://www.bilibili.com/video/BV19x411B7ZV?p=30&t=32.0
"""

import argparse

import numpy as np


def space_resection(points, f, x0, y0, m, n, convergence_threshold=3e-5, verbose=True):
    """
    Space resection by collinearity
    :param points: list of points of the form [pixel_x, pixel_y, ground_x, ground_y, ground_z]
    :param f: camera focus length
    :param x0: x0
    :param y0: y0
    :param m: denominator of the scale
    :param n: maximum number of iterations
    :param convergence_threshold: convergence threshold for the error
    :param verbose: whether to print intermediate results
    :return:
    """

    # Initialize points
    x = points[:, 0]
    y = points[:, 1]
    X = points[:, 2]
    Y = points[:, 3]
    Z = points[:, 4]

    # Initialize unknowns (guess)
    phi = omega = kappa = 0
    Xs = np.mean(X)
    Ys = np.mean(Y)
    Zs = m * f / 1000  # Est. flight altitude

    for iteration_i in range(n):
        # Residuals
        L = np.mat(np.zeros((8, 1)))  # (2 * 4) * 1 (for each point)
        # Coefficients matrix A
        A = np.mat(np.zeros((8, 6)))  # (2 * 4) * 6 (for each point)

        # Rotation matrix
        R_phi = np.mat([[np.cos(phi), 0, -np.sin(phi)],
                        [0, 1, 0],
                        [np.sin(phi), 0, np.cos(phi)]])
        R_omega = np.mat([[1, 0, 0],
                          [0, np.cos(omega), -np.sin(omega)],
                          [0, np.sin(omega), np.cos(omega)]])
        R_kappa = np.mat([[np.cos(kappa), -np.sin(kappa), 0],
                          [np.sin(kappa), np.cos(kappa), 0],
                          [0, 0, 1]])
        R = R_phi * R_omega * R_kappa

        for i in range(4):
            # For quick reference
            Xi = X[i]
            Yi = Y[i]
            Zi = Z[i]
            xi = x[i]
            yi = y[i]

            # Approximation of pixel coordinates on image plane
            x_approx = x0 - f * (
                    (R[0, 0] * (Xi - Xs) + R[1, 0] * (Yi - Ys) + R[2, 0] * (Zi - Zs)) /
                    (R[0, 2] * (Xi - Xs) + R[1, 2] * (Yi - Ys) + R[2, 2] * (Zi - Zs)))
            y_approx = y0 - f * (
                    (R[0, 1] * (Xi - Xs) + R[1, 1] * (Yi - Ys) + R[2, 1] * (Zi - Zs)) /
                    (R[0, 2] * (Xi - Xs) + R[1, 2] * (Yi - Ys) + R[2, 2] * (Zi - Zs)))

            # Residuals
            L[2 * i] = xi - x_approx
            L[2 * i + 1] = yi - y_approx

            # Coefficients matrix A
            # X_ = R[0, 0] * (Xi - Xs) + R[1, 0] * (Yi - Ys) + R[2, 0] * (Zi - Zs)
            # Y_ = R[0, 1] * (Xi - Xs) + R[1, 1] * (Yi - Ys) + R[2, 1] * (Zi - Zs)
            Zbar = R[0, 2] * (Xi - Xs) + R[1, 2] * (Yi - Ys) + R[2, 2] * (Zi - Zs)

            # ∂x/∂phi
            PxOPphi = +(yi - y0) * np.sin(omega) - (
                        ((xi - x0) / f) * ((xi - x0) * np.cos(kappa) - (yi - y0) * np.sin(kappa)) + f * np.cos(
                    kappa)) * np.cos(omega)
            # ∂x/∂omega
            PxOPomega = -f * np.sin(kappa) - ((xi - x0) / f) * ((xi - x0) * np.sin(kappa) + (yi - y0) * np.cos(kappa))
            # ∂x/∂kappa
            PxOPkappa = +(yi - y0)
            # ∂y/∂phi
            PyOPphi = +(xi - x0) * np.sin(omega) - ((yi - y0) / f) * (
                    (xi - x0) * np.cos(kappa) - (yi - y0) * np.sin(kappa) + f * np.sin(kappa)) * np.cos(omega)
            # ∂y/∂omega
            PyOPomega = -f * np.cos(kappa) - ((yi - y0) / f) * ((xi - x0) * np.sin(kappa) + (yi - y0) * np.cos(kappa))
            # ∂y/∂kappa
            PyOPkappa = -(xi - x0)

            # ∂x/∂Xs
            PxOPXs = (1 / Zbar) * (R[0, 0] * f + R[0, 2] * (xi - x0))
            # ∂x/∂Ys
            PxOPYs = (1 / Zbar) * (R[1, 0] * f + R[1, 2] * (xi - x0))
            # ∂x/∂Zs
            PxOPZs = (1 / Zbar) * (R[2, 0] * f + R[2, 2] * (xi - x0))
            # ∂y/∂Xs
            PyOPXs = (1 / Zbar) * (R[0, 1] * f + R[0, 2] * (xi - x0))
            # ∂y/∂Ys
            PyOPYs = (1 / Zbar) * (R[1, 1] * f + R[1, 2] * (xi - x0))
            # ∂y/∂Zs
            PyOPZs = (1 / Zbar) * (R[2, 1] * f + R[2, 2] * (xi - x0))

            A_ = np.mat(np.zeros((2, 6)))  # 2 * 6 sub-matrix
            A_[0, 0] = PxOPXs
            A_[0, 1] = PxOPYs
            A_[0, 2] = PxOPZs
            A_[0, 3] = PxOPphi
            A_[0, 4] = PxOPomega
            A_[0, 5] = PxOPkappa
            A_[1, 0] = PyOPXs
            A_[1, 1] = PyOPYs
            A_[1, 2] = PyOPZs
            A_[1, 3] = PyOPphi
            A_[1, 4] = PyOPomega
            A_[1, 5] = PyOPkappa

            # Merge A_ into matrix A
            A[2 * i:2 * i + 2, :] = A_

        # Solution
        X_ = (A.T * A).I * (A.T * L)

        Xs = Xs + X_[0, 0]  # Xs + ΔX
        Ys = Ys + X_[1, 0]  # Ys + ΔY
        Zs = Zs + X_[2, 0]  # Zs + ΔZ
        phi = phi + X_[3, 0]  #
        omega = omega + X_[4, 0]
        kappa = kappa + X_[5, 0]

        if verbose:
            print('Iteration: {}. ΔX = {}, ΔY = {}, ΔZ = {}, Δ𝜑 = {}, Δω = {}, Δ𝜅 = {}'.format(iteration_i, X_[1, 0],
                                                                                                 X_[2, 0], X_[3, 0],
                                                                                                 X_[3, 0], X_[4, 0],
                                                                                                 X_[5, 0]))

        if np.abs(X_[3, 0]) < convergence_threshold \
                and np.abs(X_[4, 0]) < convergence_threshold \
                and np.abs(X_[5, 0]) < convergence_threshold:
            if verbose:
                print('✓ Converged')
            break
        elif iteration_i == n - 1:
            if verbose:
                print('✕ Not converging')

    return Xs, Ys, Zs, phi, omega, kappa


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Space resection by collinearity')
    parser.add_argument('-i', '--input', type=str, help='Input control point file', required=True)
    parser.add_argument('-f', '--focus', type=float, help='Camera focus length', required=True)
    parser.add_argument('-x0', '--x0', type=float, help='x0', required=True)
    parser.add_argument('-y0', '--y0', type=float, help='y0', required=True)
    parser.add_argument('-m', '--m', type=float, help='Denominator of the scale', required=True)
    parser.add_argument('-n', '--n', type=int, help='Maximum number of iterations', required=True)
    parser.add_argument('-v', '--v', type=bool, help='Whether to print intermediate results', default=True)
    args = parser.parse_args()

    # Read input
    points = np.loadtxt(args.input)

    # Space resection
    Xs, Ys, Zs, phi, omega, kappa = space_resection(points, f=args.focus, x0=args.x0, y0=args.y0, m=args.m, n=args.n,
                                                    verbose=args.v)

    # Print result
    print('Extrinsic parameters:')
    print('Xs: {}'.format(Xs))
    print('Ys: {}'.format(Ys))
    print('Zs: {}'.format(Zs))
    print('𝜑: {}'.format(phi))
    print('ω: {}'.format(omega))
    print('𝜅: {}'.format(kappa))


if __name__ == '__main__':
    main()
