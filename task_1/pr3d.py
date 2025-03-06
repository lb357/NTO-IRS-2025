import cv2
import numpy as np
import numpy.typing as npt
import re3d
import camutil


def get_xy_from_z_pinhole(u, v, z, cameraMatrix):
    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    cx = cameraMatrix[0][2]
    cy = cameraMatrix[1][2]
    x = (u-cx)*z/fx
    y = (v-cy)*z/fy
    pos = np.array([[x], [y], [z]], dtype=np.float32)
    return pos

def apply_transform(point, M):
    x, y = point
    X = (M[0][0]*x+M[0][1]*y+M[0][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
    Y = (M[1][0]*x+M[1][1]*y+M[1][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
    return X, Y

def get_xy_from_z_perspective(u, v, z, cameraMatrix: cv2.typing.MatLike,
                             distCoeffs: cv2.typing.MatLike, sproj: float = 0.2):
    rvec, tvec = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64), np.array([0,0,0], dtype=np.float64)
    points = np.array([[-sproj, -sproj, z],
                       [sproj, -sproj, z],
                       [sproj, sproj, z],
                       [-sproj, sproj, z]], dtype=np.float32)
    plane_points = np.array([[points[0][0], points[0][1]],
                    [points[1][0], points[1][1]],
                    [points[2][0], points[2][1]],
                    [points[3][0], points[3][1]]],dtype=np.float32)
    projections, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
    pr_mat = cv2.getPerspectiveTransform(projections, plane_points)
    x, y = apply_transform([u, v], pr_mat)
    pos = np.array([[x], [y], [z]], dtype=np.float32)
    return pos, projections


if __name__ == "__main__":
    import multicam

    LCAM_ID = 2
    RCAM_ID = 0
    W, H = 1920, 1080
    MULTICAM_CALIBRATOR_ID = 80
    MULTICAM_CALIBRATOR_SIZE = 0.06
    MAX_FPS = 30

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    limg = cv2.imread('l.jpg')          # queryImage

    with open('param_left.txt') as f:
        Kl = eval(f.readline())
        Dl = eval(f.readline())

    limg = cv2.undistort(limg, Kl, Dl)
    ldimg = limg.copy()
    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)

    lcorners, lids, lrejected = detector.detectMarkers(limg)
    idx = lids[0]



    corners = lcorners[0]
    dcorners = [list(map(int, lcorners[0][0].tolist()[i])) for i in range(4)]
    lepret, leprvec, lepp = re3d.estimatePoseSingleMarkers(corners, MULTICAM_CALIBRATOR_SIZE, Kl, Dl)

    cx, cy = (dcorners[0][0]+dcorners[1][0]+dcorners[2][0]+dcorners[3][0])//4, (dcorners[0][1]+dcorners[1][1]+dcorners[2][1]+dcorners[3][1])//4

    ip, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), leprvec, lepp, Kl, Dl)
    px, py = map(int, ip[0][0])

    ip, _ = cv2.projectPoints(lepp, np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64), np.array([0,0,0], dtype=np.float64), Kl, Dl)
    zx, zy = map(int, ip[0][0])

    ppos, ccp = get_xy_from_z_perspective(cx, cy, lepp[2][0], Kl, Dl)
    for p in ccp:
        ldimg = cv2.circle(ldimg, list(map(int, p[0])), 8, (0, 255, 255), -1)

    rpos = get_xy_from_z_pinhole(cx, cy, lepp[2][0], Kl)

    print("solvePnP:", lepp.tolist())
    print("get_xy_from_z_perspective:", ppos.tolist())
    print("get_xy_from_z_pinhole:", rpos.tolist())


    #print(get_xy_from_z(cx, cy, lepp[2][0], Kl), (lepp[0][0], lepp[1][0]))

    ldimg = cv2.circle(ldimg, (cx, cy), 8, (0, 0, 255), -1)


    while cv2.waitKey(1) != ord("q"):
        cv2.imshow("DEBUG", camutil.dev_image(ldimg, 2))