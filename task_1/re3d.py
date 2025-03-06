import cv2
import numpy as np
import numpy.typing as npt

def getCTW(rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    rot_mat, jacobian_mat = cv2.Rodrigues(rvec)
    mat = np.array([
        [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], tvec[0][0]],
        [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], tvec[1][0]],
        [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], tvec[2][0]],
        [0, 0, 0, 1]
    ])
    return mat


def estimatePoseSingleMarkers(marker_points: cv2.typing.MatLike,
                              marker_size: float,
                              cameraMatrix: cv2.typing.MatLike,
                              distCoeffs: cv2.typing.MatLike) :#-> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    return cv2.solvePnP(marker_world_points, marker_points, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)

def get3D4Points(points: list, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    mat = getCTW(rvec, tvec)
    camera_points = np.array([
        np.dot(mat, points[0]),
        np.dot(mat, points[1]),
        np.dot(mat, points[2]),
        np.dot(mat, points[3])
    ])
    return camera_points[:, :-1]

def get3D3Points(points: list, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    mat = getCTW(rvec, tvec)
    camera_points = np.array([
        np.dot(mat, points[0]),
        np.dot(mat, points[1]),
        np.dot(mat, points[2])
    ])
    return camera_points[:, :-1]

def get3DMarkerCorners(marker_size: float, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    half_side = marker_size/2
    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0, 1],
                              [marker_size / 2, marker_size / 2, 0, 1],
                              [marker_size / 2, -marker_size / 2, 0, 1],
                              [-marker_size / 2, -marker_size / 2, 0, 1]], dtype=np.float32)
    return get3D4Points(marker_world_points, rvec, tvec)

def getPointsList(points: npt.ArrayLike, dtype: npt.DTypeLike = np.int16) -> list:
    return points[0].astype(dtype).tolist()

def getPointsProjection(points: npt.ArrayLike, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike,
                              cameraMatrix: cv2.typing.MatLike,
                              distCoeffs: cv2.typing.MatLike) :#-> npt.ArrayLike:
    points = np.array(points, dtype=np.float32)
    proj_points, jacobian_mat = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
    out = []
    for point in proj_points:
        point[0][0] = max(-10**10, min(point[0][0], 10**10))
        point[0][1] = max(-10**10, min(point[0][1], 10**10))
        out.append([round(point[0][0]), round(point[0][1])])
    return np.array(out)

def getLocalPointsProjection(points: npt.ArrayLike,
                             cameraMatrix: cv2.typing.MatLike,
                             distCoeffs: cv2.typing.MatLike) -> npt.ArrayLike:
    rvec = np.zeros((3, 1), np.float32) 
    tvec = np.zeros((3, 1), np.float32)
    return getPointsProjection(points, rvec, tvec, cameraMatrix, distCoeffs) 

def get3VecMat(vec: npt.ArrayLike) -> npt.ArrayLike:
    return np.array([vec[0][0], vec[1][0], vec[2][0], 1], dtype=np.float64)

def getMat3Vec(mat: npt.ArrayLike) -> npt.ArrayLike:
    return np.array([[mat[0]], [mat[1]], [mat[2]]],dtype=np.float64)