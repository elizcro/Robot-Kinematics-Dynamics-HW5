"""The paper's frame, from the four corners you touch.  THIS IS A FILE YOU EDIT AND SUBMIT.

`calibrate.py` has you touch the marker to the four corners of the sheet, in
this order, as seen standing in front of the robot looking at the sheet:

        4 ----------- 3        robot (far side)
        |             |
        |    paper    |
        |             |
        1 ----------- 2        you (near side)

Each touch gives one corner's position in the robot's base frame, in metres.
From those four points you build the paper frame P:

    origin  corner 1, moved onto the plane
    x axis  along the bottom edge, from corner 1 towards corner 2
    y axis  up the sheet, from the bottom edge towards the top edge
    z axis  the plane's unit normal, pointing up out of the paper (towards the pen)

The four touches will not lie exactly on one plane: the paper is not perfectly
flat and your hand is not perfectly steady. Use all four, as the writeup's
formula does:

    n = (p3 - p1) x (p4 - p2) / |(p3 - p1) x (p4 - p2)|     the cross product of the diagonals
    c = (p1 + p2 + p3 + p4) / 4                             the plane passes through the centroid

Fill in the three TODOs: `plane_frame`, `paper_to_robot` and `robot_to_paper`.
"""

import numpy as np


def plane_frame(corners):
    """The paper frame in the robot's base frame, as a 4x4 homogeneous transform.

    Args:
        corners: the touched corners p1, p2, p3, p4, shape (4, 3), in the order
                 drawn above.

    Returns:
        H: shape (4, 4). Its columns are [x_hat, y_hat, n, o] over [0, 0, 0, 1],
           where
             n     = the unit normal from the diagonals (see above),
             o     = p1 projected onto the plane through c with normal n,
             x_hat = (p2 - p1) with its component along n removed, normalized,
             y_hat = n cross x_hat.
    """
    corners = np.asarray(corners, dtype=float)
    p1, p2, p3, p4 = corners

    n = np.cross((p3 - p1), (p4 - p2))
    n_hat = n / np.linalg.norm(n)
    x = (p2 - p1) - (np.dot(n_hat, (p2 - p1)) * n_hat)
    x_hat = x / np.linalg.norm(x)
    y_hat = np.cross(n_hat, x_hat)

    centroid = (p1 + p2 + p3 + p4) / 4
    d = np.dot(n_hat, p1 - centroid)
    o = p1 - (d*n_hat)

    H = np.array([
        [x_hat[0], y_hat[0], n_hat[0], o[0]],
        [x_hat[1], y_hat[1], n_hat[1], o[1]],
        [x_hat[2], y_hat[2], n_hat[2], o[2]],
        [0,        0,        0,        1   ],
    ])
    return H


def paper_to_robot(H, points):
    """Points given in paper coordinates, in the robot's base frame.

    Args:
        H:      the paper frame from `plane_frame`, shape (4, 4).
        points: shape (N, 3), each row (u, v, h): u along the paper's x axis,
                v along its y axis, and h the height above the paper, all in
                metres.

    Returns:
        the same points in the robot's base frame, shape (N, 3).
    """
    points = np.asarray(points, dtype=float)

    robot_points = []
    for u, v, h in points:
        p_paper = np.array([u, v, h, 1.0])
        p_robot = np.matmul(H, p_paper)
        robot_points.append(p_robot[:3])

    return np.array(robot_points)

def robot_to_paper(H, points):
    """The inverse of `paper_to_robot`: robot-frame points, in paper coordinates.

    Args:
        H:      the paper frame from `plane_frame`, shape (4, 4).
        points: shape (N, 3), in the robot's base frame.

    Returns:
        (u, v, h) for each point, shape (N, 3).

    Hint: you do not need np.linalg.inv. The inverse of a homogeneous transform
    [R p; 0 1] is [R^T  -R^T p; 0 1].
    """
    points = np.asarray(points, dtype=float)

    R = H[:3, :3]
    o = H[:3, 3]
    R_inv = R.T
    o_inv = -np.matmul(R_inv, o)

    H_inv = np.array([
        [R_inv[0, 0], R_inv[0, 1], R_inv[0, 2], o_inv[0]],
        [R_inv[1, 0], R_inv[1, 1], R_inv[1, 2], o_inv[1]],
        [R_inv[2, 0], R_inv[2, 1], R_inv[2, 2], o_inv[2]],
        [0,           0,           0,           1       ],
    ])

    paper_points = []
    for x, y, z in points:
        p_robot = np.array([x, y, z, 1.0])
        p_paper = np.matmul(H_inv, p_robot)
        paper_points.append(p_paper[:3])

    return np.array(paper_points)
