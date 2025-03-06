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

local_ip = "0.0.0.0"
local_port = 5005

camin = input("IP: ")
camp = input("Port: ")
if camin == "":
    target_ip = "192.168.0.12"
else:
    target_ip = camin

if camp == "":
    target_port = 5005
else:
    target_port = int(camp)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((local_ip, local_port))

writer = cv2.VideoWriter("output.avi",
cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))

def send_to_robot(xr, yr, xt, yt, ang):
    message = str(xr) + " " + str(yr) + " " + str(xt) + " " + str(yt) + " " + str(ang)
    sock.sendto(message.encode('utf-8'), (target_ip, target_port))
    #print(f"Message sent to {target_ip}:{target_port}")
    #sock.settimeout(0.0)
    #try:
    #    data, addr = sock.recvfrom(1024)
    #    print(f"Received from {addr}: {data.decode('utf-8')}")
    #except socket.timeout:
    #    print("No response received within timeout.")

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
    from nto.final import Task
except Exception as err:
    pass



DEBUG = True
camin = input("Camera url: ")
if camin == "":
    CAMERA = "http://root:admin@10.128.73.80/mjpg/video.mjpg"
else:
    CAMERA = camin
#CAMERA = "rtsptextrtsp://root:admin@10.128.73.80/axis-media/media.amp?videocodec=mpeg4"
TASK_ZONE_DOWN = camutil.hsv2int(190, 20, 60)
TASK_ZONE_UP = camutil.hsv2int(210, 40,85)

#STORAGE_ZONE_DOWN = camutil.hsv2int(180, 0, 90)
#STORAGE_ZONE_UP = camutil.hsv2int(205, 25,115)
#STORAGE_INZONE_DOWN = camutil.hsv2int(40, 0, 40)
#STORAGE_INZONE_UP = camutil.hsv2int(80, 20,80)##
STORAGE_ZONE_DOWN = camutil.hsv2int(180, 0, 90)
STORAGE_ZONE_UP = camutil.hsv2int(205, 25,115)
STORAGE_INZONE_DOWN = camutil.hsv2int(20, 0, 40)
STORAGE_INZONE_UP = camutil.hsv2int(150, 20,80)##
STORAGE_ZONE_COUNT = 8



BIG_CARGOS_MODELS = {11: np.array([[-0.05, 0.10, 0.0],
                                   [0.16, 0.10, 0.0],
                                   [0.16, 0.0, 0.0],
                                   [0.05, 0.0, 0.0],
                                   [0.05, -0.205, 0.0],
                                   [-0.05, -0.205, 0.0]], dtype=np.float32)}


LOADING_ZONE_DOWN = camutil.hsv2int(300, 10, 70)
LOADING_ZONE_UP = camutil.hsv2int(360, 70,90)

UNLOADING_ZONE_DOWN = camutil.hsv2int(190, 50, 50)
UNLOADING_ZONE_UP = camutil.hsv2int(220, 100,100)

MARKER_SIZE = 0.09
UNITS = "m" #m/px
CARGO_HEIGHT_C = 10/9

ROBOT_1_IDX = 3#3
ROBOT_2_IDX = 2
SMALL_CARGOS_IDX = [4, 5, 6, 7, 8, 9, 10]
BIG_CARGOS_IDX = [11]
UNLOADING_ZONES_IDX = [0, 1]


DISPLAY_OFFSET = 4
DISPLAY_FONT_SCALE = 0.75
DISPLAY_FONT_THICKNESS = 2
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX

ROADMASK_EI = 32

DICT_4X4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(DICT_4X4, params)




with open('param.txt') as f:
    cM = eval(f.readline())
    dC = eval(f.readline())


if __name__ == "__main__":
    while True:
        try:
            task = Task()
            task.start(stage=1, task_id=int(input("task id: ")))
            task_data = task.getTask()
        except Exception:
            input("task id: ")
            task_data = [{"cargo_id": 10}, {"cargo_id": 5}]
        segmented = False
        unloading_ready = False
        print(task_data)

        video_stream = cv2.VideoCapture(CAMERA)
        while True:
            _, frame = video_stream.read()
            _, W, H = frame.shape[::-1]
            img = cv2.undistort(frame, cM, dC)
            dimg = img


            if not segmented:
                img = cv2.imread("init.png")
                task_mask = camutil.get_range_mask(img, TASK_ZONE_DOWN, TASK_ZONE_UP)
                task_mask = camutil.applyClose(task_mask, 10)
                task_mask = camutil.fillMaskContours(task_mask, 1, True)
                roadmask = task_mask.copy()

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
                    storage_mask = cv2.drawContours(storage_mask, [oc], -1, (255, 255, 255), thickness=-1)
                storage_mask = camutil.applyOpen(storage_mask, 10)
                storage_mask = camutil.fillMaskContours(storage_mask, -1, True)
                storage_contours, storage_hierarchy = cv2.findContours(storage_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                unloading_mask = camutil.get_range_mask(img, LOADING_ZONE_DOWN, LOADING_ZONE_UP)
                unloading_mask = camutil.applyClose(unloading_mask, 10)
                unloading_mask = camutil.fillMaskContours(unloading_mask, -1, True)
                unloading_mask = cv2.bitwise_and(unloading_mask, task_mask)
                unloading_mask = camutil.applyOpen(unloading_mask, 10)

                loading_mask = camutil.get_range_mask(img, UNLOADING_ZONE_DOWN, UNLOADING_ZONE_UP)
                loading_mask = camutil.applyClose(loading_mask, 10)
                loading_mask = camutil.fillMaskContours(loading_mask, -1, True)
                loading_mask = cv2.bitwise_and(loading_mask, task_mask)
                loading_mask = camutil.applyOpen(loading_mask, 10)
                segmented = True
                unloading_zones_coords = {}
                zones_coords = {}
                cargos_coords = {}
                robots_coords = {}
                small_cargos_coords = {}
                big_cargos_coords = {}

            else:
                roadmask = task_mask.copy()

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                temp_coords = {}
                for i in range(len(ids)):
                    idx = ids[i][0]
                    mcor = corners[i]
                    if UNITS == "px":
                        pixel_marker_size = (camutil.distance(*mcor[0][0], *mcor[0][1]) + camutil.distance(*mcor[0][0], *mcor[0][3])) / 2
                        marker_size = pixel_marker_size
                    elif UNITS == "m":
                        marker_size = MARKER_SIZE
                    ret, rvec, tvec = re3d.estimatePoseSingleMarkers(mcor, marker_size, cM, dC)
                    if ret:
                        marker_center, jacobian_marker_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec,
                                                                                  tvec, cM, dC)
                        wmarker_center = marker_center[0][0].astype(np.int16).tolist()
                        if idx not in UNLOADING_ZONES_IDX:
                            tvec[2][0] += CARGO_HEIGHT_C * marker_size
                        object_center, jacobian_object_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec,
                                                                                  tvec, cM, dC)
                        wobject_center = object_center[0][0].astype(np.int16).tolist()

                        mp = re3d.get3DMarkerCorners(MARKER_SIZE, rvec, tvec)
                        mpc = [(mp[0][0] + mp[1][0] + mp[2][0] + mp[3][0]) / 4,
                               (mp[0][1] + mp[1][1] + mp[2][1] + mp[3][1]) / 4]
                        mps = [(mp[0][0] + mp[1][0]) / 2, (mp[0][1] + mp[1][1]) / 2]
                        dmp = (mps[0] - mpc[0], mps[1] - mpc[1])

                        angle = -(np.arctan2(dmp[1], dmp[0]))
                        dimg = cv2.circle(dimg, wobject_center, 2, (0, 0, 255), -1)
                        dimg = cv2.circle(dimg, wmarker_center, 2, (255, 0, 0), -1)
                        if idx in BIG_CARGOS_IDX:
                            #CTW = re3d.getCTW(rvec, tvec)
                            #camera_3d_points = []
                            #for p in BIG_CARGOS_MODELS[idx]:
                            #    camera_3d_points.append(np.dot(p, CTW))
                            camera_2d_points, jacobian_mat = cv2.projectPoints(np.array(BIG_CARGOS_MODELS[idx]), rvec, tvec, cM, dC)
                            py, _, px = camera_2d_points.shape
                            camera_2d_points = camera_2d_points.reshape(1, py, px)
                            marker_data = [tvec[0][0], tvec[1][0], angle, mcor, wobject_center, camera_2d_points, tvec[2][0]]
                        else:
                            #print(getInsideContoursObj(big_cargos_mask, mcor, True))
                            marker_data = [tvec[0][0], tvec[1][0], angle, mcor, wobject_center, mcor, tvec[2][0]]

                        if idx == ROBOT_1_IDX or idx == ROBOT_2_IDX:
                            robots_coords[idx] = marker_data
                        elif idx in SMALL_CARGOS_IDX:
                            small_cargos_coords[idx] = marker_data
                        elif idx in BIG_CARGOS_IDX:
                            big_cargos_coords[idx] = marker_data
                        elif idx in UNLOADING_ZONES_IDX:
                            if not unloading_ready:
                                unloading_zones_coords[idx] = marker_data
                        else:
                            temp_coords[idx] = marker_data
                unloading_ready = True
                cargos_coords = {**small_cargos_coords, **big_cargos_coords}

                for idx in robots_coords:
                    dcorners = (robots_coords[idx][3][0].astype(np.int16).tolist())
                    x, y, w, h = cv2.boundingRect((robots_coords[idx][3][0]))
                    rect = [(x + w//2, y + h//2), round((w/2+h/2)/2*(2**(1/2)))+DISPLAY_OFFSET]
                    #dimg = camutil.dotRec(dimg, rect, (32, 128, 16), 2)
                    dimg = cv2.circle(dimg,rect[0], rect[1],(32,128,16), 2)
                    dimg = cv2.putText(dimg, str(idx), (x - 2 * DISPLAY_OFFSET, y - 2 * DISPLAY_OFFSET), DISPLAY_FONT,
                                        DISPLAY_FONT_SCALE,
                                        (32, 128, 16), DISPLAY_FONT_THICKNESS, cv2.LINE_AA)
                for idx in unloading_zones_coords:
                    dcorners = (unloading_zones_coords[idx][3][0].astype(np.int16).tolist())
                    x, y, w, h = cv2.boundingRect((unloading_zones_coords[idx][3][0]))
                    dimg = cv2.putText(dimg, str(idx),  (x- 2*DISPLAY_OFFSET, y- 2*DISPLAY_OFFSET), DISPLAY_FONT, DISPLAY_FONT_SCALE*2,
                                        (100, 100, 180), DISPLAY_FONT_THICKNESS, cv2.LINE_AA)
                for idx in cargos_coords:
                    dcorners = (cargos_coords[idx][5][0].astype(np.int16).tolist())
                    x, y, w, h = cv2.boundingRect((cargos_coords[idx][5][0]))
                    # cv2.rectangle(img, (x - DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                    #              (x + w + DISPLAY_OFFSET, y + h + DISPLAY_OFFSET), (75, 100, 150), 2, cv2.LINE_8)
                    rect = [(x - DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y + h + DISPLAY_OFFSET),
                            (x - DISPLAY_OFFSET, y + h + DISPLAY_OFFSET)]
                    if idx in BIG_CARGOS_IDX:
                        cargo_mask = np.zeros((H, W), dtype=np.uint8)
                        cargo_mask = cv2.drawContours(cargo_mask, [np.array(dcorners)], -1, (255, 255, 255), cv2.FILLED)
                        cargo_mask = applyDilate(cargo_mask, 10)
                        cargo_contours, cargo_hierarchy = cv2.findContours(cargo_mask, cv2.RETR_LIST,
                                                                                   cv2.CHAIN_APPROX_SIMPLE)
                        dimg = cv2.drawContours(dimg, cargo_contours, -1, (75, 100, 150), 3)
                    else:
                        dimg = camutil.dotRec(dimg, rect, (75, 100, 150), 2)
                    dimg = cv2.putText(dimg, str(idx),  (x- 2*DISPLAY_OFFSET, y- 2*DISPLAY_OFFSET), DISPLAY_FONT, DISPLAY_FONT_SCALE,
                                    (75, 100, 150), DISPLAY_FONT_THICKNESS, cv2.LINE_AA)
                    marker_contour = cargos_coords[idx][5].astype(int)
                    roadmask = cv2.drawContours(roadmask, [marker_contour], -1, (0, 0, 0), cv2.FILLED)

                for contour in storage_contours:
                    x, y, w, h = cv2.boundingRect((contour).astype(np.float32))
                    rect = [(x - DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y + h + DISPLAY_OFFSET),
                            (x - DISPLAY_OFFSET, y + h + DISPLAY_OFFSET)]
                    dimg = camutil.dotRec(dimg, rect, (250, 235, 200), 2, 6)

                unloading_contours,unloading_hierarchy = cv2.findContours(unloading_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in unloading_contours:
                    x, y, w, h = cv2.boundingRect((contour).astype(np.float32))
                    rect = [(x - DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y + h + DISPLAY_OFFSET),
                            (x - DISPLAY_OFFSET, y + h + DISPLAY_OFFSET)]
                    dimg = camutil.dotRec(dimg, rect, (100, 100, 180), 2, 8)

                loading_contours, loading_hierarchy = cv2.findContours(loading_mask, cv2.RETR_LIST,
                                                                           cv2.CHAIN_APPROX_SIMPLE)
                for contour in loading_contours:
                    x, y, w, h = cv2.boundingRect((contour).astype(np.float32))
                    rect = [(x - DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y - DISPLAY_OFFSET),
                            (x + w + DISPLAY_OFFSET, y + h + DISPLAY_OFFSET),
                            (x - DISPLAY_OFFSET, y + h + DISPLAY_OFFSET)]
                    dimg = camutil.dotRec(dimg, rect, (200, 110, 35), 2, 8)

                roadmask = camutil.applyErode(roadmask, ROADMASK_EI)
                roadmap = planner.get_roadmap_from_mask(roadmask, 16, (0, 0), (W, H))

                target_cargo = task_data[0]["cargo_id"]
                print("Target",target_cargo)
                if ROBOT_1_IDX in robots_coords and target_cargo in small_cargos_coords:
                    robot_x, robot_y, robot_t = robots_coords[ROBOT_1_IDX][4][0], robots_coords[ROBOT_1_IDX][4][1], robots_coords[ROBOT_1_IDX][2]
                    marker_x, marker_y = small_cargos_coords[target_cargo][4][0], small_cargos_coords[target_cargo][4][1]



                    path = np.array(nx.shortest_path(roadmap, planner.get_nearest_node([robot_x, robot_y], roadmap),
                                                     planner.get_nearest_node([marker_x, marker_y], roadmap), method="bellman-ford"))
                    path = planner.end_path([robot_x, robot_y], [marker_x, marker_y], path)

                    #for p in roadmap:
                    #    dimg = cv2.circle(dimg, p, 1, (255, 0, 0), -1)

                    path = cv2.approxPolyDP(path, 32, False)
                    path = path.reshape((len(path), 2)).tolist()
                    for i in range(1, len(path)):
                        dimg = cv2.line(dimg, path[i-1], path[i], (0, 255, 0), 3)
                    path_x, path_y = path[1]
                    real_robot, _ = pr3d.get_xy_from_z_perspective(robot_x, robot_y,
                                                                                robots_coords[ROBOT_1_IDX][6],
                                                                                cM, dC)
                    real_robot_x,real_robot_y = real_robot[0][0], real_robot[1][0]
                    real_path, _ = pr3d.get_xy_from_z_perspective(path_x, path_y,
                                                                                robots_coords[ROBOT_1_IDX][6],
                                                                                cM, dC)
                    real_path_x, real_path_y = real_path[0][0], real_path[1][0]
                    #print(camutil.distance(real_robot_x, real_robot_y,real_path_x,real_path_y))
                    if camutil.distance(real_robot_x, real_robot_y,real_path_x,real_path_y) > 0.03:
                        #print(real_robot_x, real_robot_y,real_path_x,real_path_y,robots_coords[ROBOT_1_IDX][2])
                        #t = threading.Thread(send_to_robot, args=(real_robot_x, real_robot_y,real_path_x,real_path_y,robots_coords[ROBOT_1_IDX][2]))
                        #t.start()
                        #t.join()
                        send_to_robot(real_robot_x, real_robot_y,real_path_x,real_path_y,robots_coords[ROBOT_1_IDX][2])
                    else:
                        task_data = task_data[1:]
            if DEBUG:
                cv2.imshow("dimg", camutil.dev_image(dimg, 1.5))
                writer.write(dimg)
                b = cv2.waitKey(1)
                if b == ord("e"):
                    break
                if b == ord("q"):
                    try:
                        task.stop()
                    except Exception:
                        pass
                    writer.release()
                    exit()
    video_stream.release()

