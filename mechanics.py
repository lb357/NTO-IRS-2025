import math


def get_collider(marker_x, marker_y, target_x, target_y, r):
    angle = math.atan2(target_x-marker_x, target_y-marker_y) + math.pi
    ofx = math.cos(angle)*r
    ofy = math.sin(angle)*r
    return marker_x+ofx, marker_y + ofy
