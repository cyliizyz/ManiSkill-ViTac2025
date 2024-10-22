import numpy as np
import transforms3d as t3d
import fcl


def inv_trans_np(pos, position, rotation):
    """compute the relative pos to the object"""
    R = t3d.quaternions.quat2mat(rotation)
    return R.T @ (pos - position)


def qrot_np(q, v):
    return t3d.quaternions.rotate_vector(v, q)


def transform_pts(pts, RT):
    n = pts.shape[0]
    pts = np.concatenate([pts, np.ones((n, 1))], axis=1)
    pts = RT @ pts.T
    pts = pts.T[:, :3]
    return pts


def dist2np(pts_0, pts_1):
    """compute MxN point distance"""
    square_sum0 = np.sum(pts_0 ** 2, axis=1, keepdims=True)
    square_sum1 = np.sum(pts_1 ** 2, axis=1, keepdims=True)
    square_sum = square_sum0 + square_sum1.T
    square_sum -= 2 * pts_0 @ pts_1.T
    return np.sqrt(square_sum + 1e-7)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


def quat2R_np(q):
    """quaternion to rotation matrix"""
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array(
        [
            [2 * (w * w + x * x) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w * w + y * y) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w * w + z * z) - 1],
        ]
    )
    return R


def estimate_rigid_transform(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = np.linalg.det(V) * np.linalg.det(W)

    D = np.eye(3)
    D[2, 2] = d

    R = np.dot(np.dot(V, D), W)

    t = Q.mean(axis=0) - P.mean(axis=0).dot(R)

    return R, t


def generate_offset(xy_range, r_min, theta_range):
    offset_xy = np.zeros(2)
    while offset_xy[0] ** 2 + offset_xy[1] ** 2 <= r_min ** 2:
        offset_xy = np.random.rand(2) * 2 * xy_range + np.array([-xy_range])

    offset = np.zeros(3)
    offset[0] = offset_xy[0]
    offset[1] = offset_xy[1]
    offset[2] = (
            np.random.rand(1) * 2 * theta_range / 180 * np.pi - theta_range / 180 * np.pi
    )
    return offset


def generate_mono_offset(xy_max, xy_min):
    if np.random.rand(1) > 0.5:
        x = np.random.rand(1)[0] * 2 * (xy_max - xy_min) - (xy_max - xy_min)
        if x >= 0:
            x += xy_min
        else:
            x -= xy_min
        y = 0
    else:
        y = np.random.rand(1)[0] * 2 * (xy_max - xy_min) - (xy_max - xy_min)
        if y >= 0:
            y += xy_min
        else:
            y -= xy_min
        x = 0

    offset = np.array([x, y, 0])
    return offset


def generate_blocked_offset(pos_offset_range, rot_offset_range, peg_v, peg_f, hole_v, hole_f):
    peg = fcl.BVHModel()
    peg.beginModel(peg_v.shape[0], peg_f.shape[0])
    peg.addSubModel(peg_v, peg_f)
    peg.endModel()

    hole = fcl.BVHModel()
    hole.beginModel(hole_v.shape[0], hole_f.shape[0])
    hole.addSubModel(hole_v, hole_f)
    hole.endModel()

    t1 = fcl.Transform()
    peg_fcl = fcl.CollisionObject(peg, t1)

    t2 = fcl.Transform()
    hole_fcl = fcl.CollisionObject(hole, t2)
    while True:  # loop until generate valid offset
        x_offset, y_offset = (np.random.rand(2) * 2 - 1) * pos_offset_range / 1000.0
        theta_offset = (np.random.rand() * 2 - 1) * rot_offset_range * np.pi / 180
        R = t3d.euler.euler2mat(0.0, 0.0, theta_offset, axes="rxyz")
        T = np.array([x_offset, y_offset, 0.0])

        t3 = fcl.Transform(R, T)
        peg_fcl.setTransform(t3)
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        ret = fcl.collide(peg_fcl, hole_fcl, request, result)
        if ret > 0:
            offset = np.array([x_offset * 1000, y_offset * 1000, theta_offset * 180 / np.pi])
            break
    return offset


def EulerToQuternion(euler):
    """
        Euler to Quternion
    """
    euler = euler * np.pi / 180
    x1 = np.cos(euler[1]) * np.cos(euler[2])
    x2 = np.cos(euler[1]) * np.sin(euler[2])
    x3 = -np.sin(euler[1])

    y1 = -np.cos(euler[0]) * np.sin(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
    y2 = np.cos(euler[0]) * np.cos(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
    y3 = np.sin(euler[0]) * np.cos(euler[1])

    z1 = np.sin(euler[0]) * np.sin(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
    z2 = -np.sin(euler[0]) * np.cos(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
    z3 = np.cos(euler[0]) * np.cos(euler[1])

    Q = np.zeros(4, dtype=float)
    Q[0] = np.sqrt(x1 + y2 + z3 + 1) / 2
    if y3 > z2:
        Q[1] = np.sqrt(x1 - y2 - z3 + 1) / 2
    else:
        Q[1] = -np.sqrt(x1 - y2 - z3 + 1) / 2

    if z1 > x3:
        Q[2] = np.sqrt(y2 - x1 - z3 + 1) / 2
    else:
        Q[2] = -np.sqrt(y2 - x1 - z3 + 1) / 2

    if x2 > y1:
        Q[3] = np.sqrt(z3 - x1 - y2 + 1) / 2
    else:
        Q[3] = -np.sqrt(z3 - x1 - y2 + 1) / 2
    return Q

if __name__ == "__main__":
    for i in range(10):
        q = np.random.rand(4)
        q /= np.linalg.norm(q)
        R_np = quat2R_np(q)
        R_3d = t3d.quaternions.quat2mat(q)
        print(np.allclose(R_np, R_3d))
