import cv2
import camutil
from camutil import *
import numpy as np
import re3d
import pr3d
import planner
import socket
import networkx as nx
import threading
import time
from smart_input import sinput
from dotenv import load_dotenv
load_dotenv(".env")
from nto.final import Task
import mechanics


ROBOT_IP = sinput("Robot ip: ", "192.168.0.12")
ROBOT_PORT = int(sinput("Robot port: ", "5005"))
CAMERA = sinput("Camera url: ", "http://root:admin@10.128.73.80/mjpg/video.mjpg")

TASK_ZONE_DOWN = camutil.hsv2int(190, 20, 60)
TASK_ZONE_UP = camutil.hsv2int(210, 40,85)

STORAGE_ZONE_DOWN = camutil.hsv2int(180, 0, 90)
STORAGE_ZONE_UP = camutil.hsv2int(205, 25,115)
STORAGE_INZONE_DOWN = camutil.hsv2int(20, 0, 40)
STORAGE_INZONE_UP = camutil.hsv2int(150, 20,80)
STORAGE_ZONE_COUNT = 8

BIG_CARGOS_MODELS = {11: np.array([[-0.05, 0.10, 0.0],
                                   [0.16, 0.10, 0.0],
                                   [0.16, 0.0, 0.0],
                                   [0.05, 0.0, 0.0],
                                   [0.05, -0.205, 0.0],
                                   [-0.05, -0.205, 0.0]], dtype=np.float32)}


UNLOADING_ZONES_IDX = [0, 1]
SMALL_CARGOS_IDX = [4, 5, 6, 7, 8, 9, 10]
ROBOT_1_IDX = 3#3
ROBOT_2_IDX = 2
BIG_CARGOS_IDX = [11]

MARKER_SIZE = 0.09
CARGO_HEIGHT_C = 10/9


UNITS = "m" #m/px

ROADMASK_EI = 32
ROADMASK_OR = 40



DICT_4X4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(DICT_4X4, params)


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

segmented = False

with open('param.txt') as f:
    cM = eval(f.readline())
    dC = eval(f.readline())

def send_to_robot(xr, yr, xt, yt, ang, sock, target_ip, target_port):
    message = str(xr) + " " + str(yr) + " " + str(xt) + " " + str(yt) + " " + str(ang)
    sock.sendto(message.encode('utf-8'), (target_ip, target_port))

def mass_center(contour, is_rounded: bool = True):
    moments = cv2.moments(np.array(contour, dtype=np.float32))
    if moments['m00'] != 0:
        x = float(moments['m10'] / moments['m00'])
        y = float(moments['m01'] / moments['m00'])
        if is_rounded:
            return round(x), round(y)
        else:
            return x, y
    else:
        return None

task = Task()
try:
    task.start(stage=2, task_id=int(sinput("Task id: ", "1")))
    task_json = task.getTask()
except Exception as err:
    print(err)
    task_json = [{"storage_zone": "", "cargo_id": 7}]
#task_json = [{"unloading_zone": "42", "cargo_id": 18}]
print(task_json)
video_stream = cv2.VideoCapture(CAMERA)

while True:
    if not segmented:
        _, frame = video_stream.read()
        _, W, H = frame.shape[::-1]
        unloading_zones = {}
        storage_zones = {}
        small_cargos = {}
        big_cargos = {}
        robots_coords = {}
        img = cv2.imread("init.png")
        task_mask = camutil.get_range_mask(img, TASK_ZONE_DOWN, TASK_ZONE_UP)
        task_mask = camutil.applyClose(task_mask, 10)
        task_mask = camutil.fillMaskContours(task_mask, 1, True)
        #task_mask = (np.ones((H, W), dtype=np.uint8)*255).astype(np.uint8)

        storage_outmask = camutil.get_range_mask(img, STORAGE_ZONE_DOWN, STORAGE_ZONE_UP)
        storage_inmask = camutil.get_range_mask(img, STORAGE_INZONE_DOWN, STORAGE_INZONE_UP)
        storage_inmask = camutil.applyOpen(storage_inmask, 1)
        storage_inmask = camutil.applyClose(storage_inmask, 1)
        storage_outmask = camutil.applyClose(storage_outmask, 10)
        storage_outmask = camutil.applyErode(storage_outmask, 1)
        storage_outmask = camutil.fillMaskContours(storage_outmask, -1, False)
        storage_contours_est = camutil.getInsideContours(storage_outmask, storage_inmask, True, 2)
        storage_mask = np.zeros(storage_outmask.shape, dtype=np.uint8)
        for oc, ic in storage_contours_est:
            oc = oc.reshape((len(oc), 2))
            storage_mask = cv2.drawContours(storage_mask, [oc], -1, (255, 255, 255), thickness=-1)
        storage_mask = camutil.applyOpen(storage_mask, 10)
        storage_mask = camutil.fillMaskContours(storage_mask, -1, True)
        storage_contours, storage_hierarchy = cv2.findContours(storage_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(storage_contours)):
            storage_zones[i] = mass_center(storage_contours[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        for i in range(len(ids)):
            idx = ids[i][0]
            mcor = corners[i]
            if idx in UNLOADING_ZONES_IDX:
                unloading_zones[idx] = mass_center(mcor)
        segmented = True
        storage_mask_ = storage_mask.copy()
        roadmask_robot = task_mask.copy()
    else:
        storage_mask = storage_mask_.copy()
        _, frame = video_stream.read()
        _, W, H = frame.shape[::-1]
        img = cv2.undistort(frame, cM, dC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for i in range(len(ids)):
                idx = ids[i][0]
                mcor = corners[i]
                if idx in SMALL_CARGOS_IDX:
                    small_cargos[idx] = mass_center(mcor)
                elif idx == ROBOT_1_IDX or idx == ROBOT_2_IDX:
                    ret, rvec, tvec = re3d.estimatePoseSingleMarkers(mcor, MARKER_SIZE, cM, dC)
                    if ret:
                        marker_center, jacobian_marker_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec,
                                                                                  tvec, cM, dC)
                        wmarker_center = marker_center[0][0].astype(np.int16).tolist()
                        if idx not in UNLOADING_ZONES_IDX:
                            tvec[2][0] += CARGO_HEIGHT_C * MARKER_SIZE
                        object_center, jacobian_object_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec,
                                                                                  tvec, cM, dC)
                        wobject_center = object_center[0][0].astype(np.int16).tolist()

                        mp = re3d.get3DMarkerCorners(MARKER_SIZE, rvec, tvec)
                        mpc = [(mp[0][0] + mp[1][0] + mp[2][0] + mp[3][0]) / 4,
                               (mp[0][1] + mp[1][1] + mp[2][1] + mp[3][1]) / 4]
                        mps = [(mp[0][0] + mp[1][0]) / 2, (mp[0][1] + mp[1][1]) / 2]
                        dmp = (mps[0] - mpc[0], mps[1] - mpc[1])

                        angle = -(np.arctan2(dmp[1], dmp[0]))

                        robots_coords[idx] = [tvec[0][0], tvec[1][0], angle, wobject_center, tvec[2][0]]
                elif idx in BIG_CARGOS_IDX:
                    big_cargos[idx] = mass_center(mcor)
                    ret, rvec, tvec = re3d.estimatePoseSingleMarkers(mcor, MARKER_SIZE, cM, dC)
                    if ret:
                        camera_2d_points, jacobian_mat = cv2.projectPoints(np.array(BIG_CARGOS_MODELS[idx]),
                                                                           rvec, tvec, cM, dC)
                        py, _, px = camera_2d_points.shape
                        camera_2d_points = camera_2d_points.reshape(py, px, 1)
                        camera_2d_points = camera_2d_points.astype(np.int16)
                        camera_2d_points = camera_2d_points.reshape((len(camera_2d_points), 2))
                        for p in camera_2d_points:
                            task_mask = cv2.circle(task_mask, p, round(ROADMASK_OR*1), (0, 0, 0), -1)


        if len(task_json) == 0:
            send_to_robot(0, 0, 0, 0, 0, sock, ROBOT_IP, ROBOT_PORT)
            print("Finish!")
            break
        else:
            cargo_idx = int(task_json[0]["cargo_id"])
            if "unloading_zone" in task_json[0]:
                zone_idx = int(task_json[0]["unloading_zone"])
                zone_with_idx = True
            else:
                zone_idx = min(storage_zones.keys())
                zone_with_idx = False

        if (cargo_idx in small_cargos) and (zone_idx in unloading_zones or zone_idx in storage_zones) :
            cargo_center = small_cargos[cargo_idx]
            if zone_with_idx:
                zone_center = unloading_zones[zone_idx]
            else:
                zone_center = storage_zones[zone_idx]

            roadmask_cargo = task_mask.copy()
            roadmask_robot = task_mask.copy()
            for cargo in small_cargos:
                if cargo != cargo_idx:
                    roadmask_cargo = cv2.circle(roadmask_cargo, small_cargos[cargo], ROADMASK_OR, (0, 0, 0), -1)
                roadmask_robot = cv2.circle(roadmask_robot, small_cargos[cargo], round(ROADMASK_OR*1.25), (0, 0, 0), -1)
            roadmap_cargo = planner.get_roadmap_from_mask(roadmask_cargo, 16, (0, 0), (W, H))
            roadmap_robot = planner.get_roadmap_from_mask(roadmask_robot, 16, (0, 0), (W, H))

            cargo_path = np.array(nx.shortest_path(roadmap_cargo, planner.get_nearest_node(cargo_center, roadmap_cargo),
                                             planner.get_nearest_node(zone_center, roadmap_cargo),
                                             method="bellman-ford"))
            cargo_path = planner.end_path(cargo_center, zone_center, cargo_path)
            cargo_path = cv2.approxPolyDP(cargo_path, 32, False)
            cargo_path = cargo_path.reshape((len(cargo_path), 2)).tolist()

            ##D
            #for i in range(1, len(cargo_path)):
            #    #print(cargo_path)
            #    img = cv2.line(img, cargo_path[i - 1], cargo_path[i], (0, 255, 255), 3)

            cargo_path_x, cargo_path_y = cargo_path[1]
            #print(*cargo_center, cargo_path_x, cargo_path_y)
            collider_x, collider_y = map(int, mechanics.get_collider(*cargo_center, cargo_path_x, cargo_path_y, ROADMASK_OR))

            if ROBOT_1_IDX in robots_coords:
                robot_center = robots_coords[ROBOT_1_IDX][3]
                collider_center = [collider_x, collider_y]
                robot_path = np.array(nx.shortest_path(roadmap_robot, planner.get_nearest_node(robot_center, roadmap_robot),
                                                       planner.get_nearest_node(collider_center, roadmap_robot),
                                                       method="bellman-ford"))
                robot_path = planner.end_path(robot_center, collider_center, robot_path)
                robot_path = cv2.approxPolyDP(robot_path, 8, False)
                robot_path = robot_path.reshape((len(robot_path), 2)).tolist()
                # D
                for i in range(1, len(robot_path)):
                    # print(cargo_path)
                    img = cv2.line(img, robot_path[i - 1], robot_path[i], (255, 64, 64), 3)

                real_robot, _ = pr3d.get_xy_from_z_perspective(*robot_center,
                                                               robots_coords[ROBOT_1_IDX][4],
                                                               cM, dC)
                real_robot_x, real_robot_y = real_robot[0][0], real_robot[1][0]
                if cv2.arcLength(robot_path) <= ROADMASK_OR*1.5:
                    real_path, _ = pr3d.get_xy_from_z_perspective(*cargo_center,
                                                                  robots_coords[ROBOT_1_IDX][4],
                                                                  cM, dC)
                else:
                    real_path, _ = pr3d.get_xy_from_z_perspective(*robot_path[1],
                                                              robots_coords[ROBOT_1_IDX][4],
                                                              cM, dC)

                real_path_x, real_path_y = real_path[0][0], real_path[1][0]

                robot_angle = robots_coords[ROBOT_1_IDX][2]

                #print(real_robot_x, real_robot_y,real_path_x, real_path_y, robot_angle)
                send_to_robot(real_robot_x, real_robot_y,real_path_x, real_path_y, robot_angle, sock, ROBOT_IP, ROBOT_PORT)


                if camutil.distance(*cargo_center, *zone_center) <= ROADMASK_OR/1.5:
                    task_json = task_json[1:]

            #D
            img = cv2.circle(img, [collider_x, collider_y], 5, (0, 0, 255), -1)
            img = cv2.circle(img, zone_center, 5, (0, 0, 255), -1)
            if ROBOT_1_IDX in robots_coords:
                img = cv2.circle(img, robots_coords[ROBOT_1_IDX][3], 6, (255, 0, 255), -1)

        #for p in roadmap_cargo:
        #    imh=cv2.circle(img, p, 1, (255, 0, 0), -1)
        #for p in roadmap_robot:
        #    imh=cv2.circle(img, p, 1, (0, 255, 0), -1)
        cv2.imshow("d", img)
        cv2.imshow("s", roadmask_robot)
        #print(robots_coords)
        cv2.waitKey(1)