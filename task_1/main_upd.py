import cv2
import camutil
import numpy as np
import re3d

def get_range_mask(img, down, up):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, down, up)
    return mask

def applyClose(mask, i=1, k=3):
    mask_kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mask_kernel, iterations=i)

def applyOpen(mask, i=1, k=3):
    mask_kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, mask_kernel, iterations=i)

def fill(mask, is_hull = False):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if is_hull:
            contour = cv2.convexHull(contour)
        mask = cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=-1)
    return mask

def fillMaskContours(mask, i = -1, is_hull = False):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data = {}
    for contour in contours:
        S = float(cv2.contourArea(contour))
        if S not in data:
            data[S] = []
        data[S].append(contour)
    mask = np.zeros(mask.shape, dtype=np.uint8)
    l = sorted(data.keys())[::-1]
    out = []
    if i == -1:
        i = len(data)

    while i > 0 and len(l) > 0:
        out.append(data[float(l[0])][0])
        l = l[1:]
        i -= 1

    for k in out:
        if is_hull:
            k = cv2.convexHull(k)
        mask = cv2.drawContours(mask, [k], -1, (255, 255, 255), thickness=-1)
    return mask

def getInsideContours(outside_mask, inside_mask, use_flag = False, min_s = 0):
    outside_contours, ohierarchy = cv2.findContours(outside_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inside_contours, ihierarchy = cv2.findContours(inside_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data = []
    for oc in outside_contours:
        for ic in inside_contours:
            if oc is not ic and cv2.contourArea(ic, min_s) >= min_s:
                if cv2.pointPolygonTest(oc, np.array(ic[0][0], dtype=np.float64), False) != -1:
                    data.append([oc, ic])
                    if use_flag:
                        break
    return data


DEBUG = True
CAMERA = "test.jpg"

TASK_ZONE_DOWN = camutil.hsv2int(190, 20, 60)
TASK_ZONE_UP = camutil.hsv2int(210, 40,85)

STORAGE_ZONE_DOWN = camutil.hsv2int(180, 0, 90)
STORAGE_ZONE_UP = camutil.hsv2int(205, 25,115)
STORAGE_INZONE_DOWN = camutil.hsv2int(40, 0, 40)
STORAGE_INZONE_UP = camutil.hsv2int(80, 20,80)##
STORAGE_ZONE_COUNT = 8

LOADING_ZONE_DOWN = camutil.hsv2int(290, 10, 70)
LOADING_ZONE_UP = camutil.hsv2int(360, 70,90)

UNLOADING_ZONE_DOWN = camutil.hsv2int(190, 50, 50)
UNLOADING_ZONE_UP = camutil.hsv2int(220, 100,100)

MARKER_SIZE = 0.02
CARGO_HEIGHT = 0.02


DICT_4X4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(DICT_4X4, params)

segmented = False

with open('param.txt') as f:
    cM = eval(f.readline())
    dC = eval(f.readline())


if __name__ == "__main__":
    while True:
        frame = cv2.imread("test.jpg")
        img = cv2.undistort(frame, cM,dC)
        if not segmented:
            task_mask = get_range_mask(img, TASK_ZONE_DOWN, TASK_ZONE_UP)
            task_mask = applyClose(task_mask, 10)
            task_mask = fillMaskContours(task_mask, 1, True)

            storage_outmask = get_range_mask(img, STORAGE_ZONE_DOWN, STORAGE_ZONE_UP)
            storage_inmask = get_range_mask(img, STORAGE_INZONE_DOWN, STORAGE_INZONE_UP)
            storage_outmask = applyClose(storage_outmask, 10)
            storage_outmask = fillMaskContours(storage_outmask, -1, True)
            storage_contours_est = getInsideContours(storage_outmask, storage_inmask, True)
            storage_mask = np.zeros(storage_outmask.shape, dtype=np.uint8)
            for oc, ic in storage_contours_est:
                storage_mask = cv2.drawContours(storage_mask, [oc], -1, (255, 255, 255), thickness=-1)
            storage_mask = applyOpen(storage_mask, 10)

            loading_mask = get_range_mask(img, LOADING_ZONE_DOWN, LOADING_ZONE_UP)
            loading_mask = applyClose(loading_mask, 10)
            loading_mask = fillMaskContours(loading_mask, -1, True)
            loading_mask = cv2.bitwise_and(loading_mask, task_mask)
            loading_mask = applyOpen(loading_mask, 10)

            unloading_mask = get_range_mask(img, UNLOADING_ZONE_DOWN, UNLOADING_ZONE_UP)
            unloading_mask = applyClose(unloading_mask, 10)
            unloading_mask = fillMaskContours(unloading_mask, -1, True)
            unloading_mask = cv2.bitwise_and(unloading_mask, task_mask)
            unloading_mask = applyOpen(unloading_mask, 10)
            segmented = True
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            for i in range(len(ids)):
                idx = ids[i]
                mcor = corners[i]
                if True:
                    ret, rvec, tvec = re3d.estimatePoseSingleMarkers(mcor, MARKER_SIZE, cM, dC)
                    if ret:
                        #tvec[2][0] += CARGO_HEIGHT
                        marker_center, jacobian_marker_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec,
                                                                                  tvec, cM, dC)
                        wmarker_center = marker_center[0][0].astype(np.int16).tolist()

                        mp = re3d.get3DMarkerCorners(MARKER_SIZE, rvec, tvec)
                        mpc = [(mp[0][0] + mp[1][0] + mp[2][0] + mp[3][0]) / 4,
                               (mp[0][1] + mp[1][1] + mp[2][1] + mp[3][1]) / 4]
                        mps = [(mp[1][0] + mp[2][0]) / 2, (mp[1][1] + mp[2][1]) / 2]
                        dmp = (mps[0] - mpc[0], mps[1] - mpc[1])

                        angle = -np.degrees(np.arctan2(dmp[1], dmp[0]))
                        print(idx, tvec, angle)
                        img = cv2.circle(img, wmarker_center, 6, (255, 0, 0), -1)

        if DEBUG:
            cv2.imshow("DEBUG", camutil.dev_image(storage_mask, 2))
            cv2.imshow("DEBUG4", camutil.dev_image(unloading_mask, 2))
            cv2.imshow("DEBUG3", camutil.dev_image(loading_mask, 2))
            cv2.imshow("DEBUG2", camutil.dev_image(img, 2))
            cv2.waitKey(1)
    video_stream.release()