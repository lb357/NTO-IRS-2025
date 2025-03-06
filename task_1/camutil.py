import cv2
import numpy as np

def dev_image(image: cv2.typing.MatLike, k: float) -> cv2.typing.MatLike:
    return cv2.resize(image.copy(), (round(image.shape[1]/k), round(image.shape[0]/k)))

def overlay(background, img, x, y):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b = np.copy(background)
    
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))

    place = b[y: y + img.shape[0], x: x + img.shape[1]]
    a = img[..., 3:].repeat(3, axis=2).astype('uint16')
    place[..., :3] = (place[..., :3].astype('uint16') * (255 - a) // 255) + img[..., :3].astype('uint16') * a // 255
    return cv2.cvtColor(b, cv2.COLOR_RGBA2BGR)


def hsv2int(h, s, v):
    return h / 360 * 179, s / 100 * 255, v / 100 * 255


def int2hsv(h, s, v):
    return h / 179 * 360, s / 255 * 100, v / 255 * 100

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2))

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

def applyErode(mask, i=1, k=3):
    mask_kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.erode(mask, mask_kernel, iterations=i)

def applyDilate(mask, i=1, k=3):
    mask_kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask, mask_kernel, iterations=i)

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

def getInsideContoursObj(outside_mask, inside_contours, use_flag = False, min_s = 0):
    outside_contours, ohierarchy = cv2.findContours(outside_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data = []
    for oc in outside_contours:
        for ic in inside_contours:
            if oc is not ic and cv2.contourArea(ic) >= min_s:
                if cv2.pointPolygonTest(oc, np.array(ic[0], dtype=np.float64), False) != -1:
                    data.append([oc, ic])
                    if use_flag:
                        break
    return data

def getInsideContours(outside_mask, inside_mask, use_flag = False, min_s = 0):
    outside_contours, ohierarchy = cv2.findContours(outside_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inside_contours, ihierarchy = cv2.findContours(inside_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    data = []
    for oc in outside_contours:
        for ic in inside_contours:
            if oc is not ic and cv2.contourArea(ic) >= min_s:
                if cv2.pointPolygonTest(oc, np.array(ic[0][0], dtype=np.float64), False) != -1:
                    data.append([oc, ic])
                    if use_flag:
                        break
    return data

def dotRec(img, points, color, th = 1, k = 8):
    points.append(points[0])
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        for j in range(1, k, 2):
            cv2.line(img, [points[i-1][0]+(j)*(dx//k), points[i-1][1]+(j)*(dy//k)],
                     [points[i - 1][0] + (j-1) * (dx // k), points[i - 1][1] + (j-1) * (dy // k)],
                     color, th)
    return img