import networkx as nx
import numpy as np
import numpy.typing as npt
import multiprocessing.pool


def get_roadmap_from_mask(road_mask: npt.ArrayLike, step: int,
                          start_point: tuple = (0, 0), end_point: tuple = (0, 0)) -> nx.Graph:
    roadmap = nx.Graph()
    for y in range(start_point[1], end_point[1], step):
        for x in range(start_point[0], end_point[0], step):
            if road_mask[y][x] != 0:
                p = (x, y)
                if not roadmap.has_node(p):
                    roadmap.add_node(p)

                if y + step < start_point[1]:
                    if road_mask[y + step][x]:
                        t = (x, y + step)
                        if not roadmap.has_node(t):
                            roadmap.add_node(t)
                        roadmap.add_edge(p, t)
                if x + step < start_point[0]:
                    if road_mask[y][x + step]:
                        t = (x + step, y)
                        if not roadmap.has_node(t):
                            roadmap.add_node(t)
                        roadmap.add_edge(p, t)
                if y - step >= 0:
                    if road_mask[y - step][x]:
                        t = (x, y - step)
                        if not roadmap.has_node(t):
                            roadmap.add_node(t)
                        roadmap.add_edge(p, t)
                if x - step >= 0:
                    if road_mask[y][x - step]:
                        t = (x - step, y)
                        if not roadmap.has_node(t):
                            roadmap.add_node(t)
                        roadmap.add_edge(p, t)
    return roadmap


def get_roadmap_from_mask_chunk(road_mask: npt.ArrayLike, step: int,
                                start_point: tuple = (0, 0), end_point: tuple = (0, 0),
                                chunk_size: int = 32) :#-> nx.Graph:
    roadmap = nx.Graph()
    for y in range(start_point[1], end_point[1], chunk_size):
        for x in range(start_point[0], end_point[0], chunk_size):
            if x + chunk_size < end_point[0]:
                mx = x + chunk_size
            else:
                mx = end_point[0]
            if y + chunk_size < end_point[1]:
                my = y + chunk_size
            else:
                my = end_point[1]
            chunk = get_roadmap_from_mask(road_mask, step, (x, y), (mx, my))
            roadmap = nx.compose(roadmap, chunk)
    return roadmap


def get_roadmap_from_mask_chunk_parallel(road_mask: npt.ArrayLike, step: int,
                                         start_point: tuple = (0, 0), end_point: tuple = (0, 0),
                                         chunk_size: int = 32, pool: [multiprocessing.pool.Pool,
                                                                      multiprocessing.pool.ThreadPool] = None) :#-> nx.Graph:
    if pool is None:
        pool = create_planner_pool()

    pool_results = []
    roadmap = nx.Graph()
    for y in range(start_point[1], end_point[1], chunk_size):
        for x in range(start_point[0], end_point[0], chunk_size):
            if x + chunk_size < end_point[0]:
                mx = x + chunk_size
            else:
                mx = end_point[0]
            if y + chunk_size < end_point[1]:
                my = y + chunk_size
            else:
                my = end_point[1]
            pool_results.append(pool.apply_async(get_roadmap_from_mask, (road_mask, step, (x, y), (mx, my))))
    for result in pool_results:
        roadmap = nx.compose(roadmap, result.get())
    return roadmap


def create_planner_pool(max_processes: int = 8, use_threads: bool = False) :#-> multiprocessing.pool.Pool:
    if use_threads:
        pool = multiprocessing.pool.ThreadPool(processes=max_processes)
    else:
        pool = multiprocessing.pool.Pool(processes=max_processes)
    return pool


def get_nearest_node(point: tuple, graph: nx.Graph) -> tuple:
    np_graph = np.array(graph.nodes)
    np_point = np.array(point)
    distances = np.linalg.norm(np_graph - np_point, axis=1)
    min_index = np.argmin(distances)
    return tuple(np_graph[min_index])


def get_roadmap_from_size(size: tuple = (1920, 1080), step: int = 12,
                          start_point: tuple = (0, 0), end_point: tuple = (0, 0)):# -> nx.Graph:
    roadmap = nx.Graph()
    for y in range(start_point[1], end_point[1], step):
        for x in range(start_point[0], end_point[0], step):
            p = (x, y)
            roadmap.add_node(p)
            if y + step < start_point[1]:
                t = (x, y + step)
                roadmap.add_node(t)
                roadmap.add_edge(p, t)
            if x + step < start_point[0]:
                t = (x + step, y)
                roadmap.add_node(t)
                roadmap.add_edge(p, t)
            if y - step >= 0:
                t = (x, y - step)
                roadmap.add_node(t)
                roadmap.add_edge(p, t)
            if x - step >= 0:
                t = (x - step, y)
                roadmap.add_node(t)
                roadmap.add_edge(p, t)
    return roadmap


def clear_roadmap_from_mask(mask: npt.ArrayLike, roadmap: nx.Graph) -> nx.Graph:
    for point in list(roadmap.nodes.keys()):
        if mask[point[1]][point[0]] == 0:
            roadmap.remove_node(point)
    return roadmap


def clear_roadmap_zone(zones: list,
                        roadmap: nx.Graph) -> nx.Graph:
    for point in list(roadmap.nodes.keys()):
        for zone in zones:
            start_point = zone[0]
            end_point = zone[1]
            if start_point[0] <= point[0] <= end_point[0] and start_point[1] <= point[1] <= end_point[1]:
                roadmap.remove_node(point)
    return roadmap


def end_path(from_point: list, to_point: list, path: npt.ArrayLike):
    if path[0].tolist() != from_point:
        path = np.array([from_point, *path])
    if path[-1].tolist() != to_point:
        path = np.array([*path, to_point])
    return path


if __name__ == "__main__":
    import cv2
    import time

    while cv2.waitKey(1) != ord("q"):
        mask = np.full([1080, 1920], 255, dtype=np.uint8)
        mask = cv2.circle(mask, (64, 64), 16, (0, 0, 0), -1)
        mask = cv2.circle(mask, (100, 64), 18, (0, 0, 0), -1)
        mask = cv2.circle(mask, (120, 72), 16, (0, 0, 0), -1)

        tms = time.perf_counter()
        #rm = get_roadmap_from_mask(mask, 12, (0, 0), (1920, 1080))

        rm = get_roadmap_from_size((1920, 1080), 12, (0, 0), (1920, 1080))
        rm = clear_roadmap_from_mask(mask, rm)
        rm = clear_roadmap_zone([((80, 80), (100, 100))], rm)

        tme = time.perf_counter()
        print(f"Generate roadmap: {tme - tms}")
        tms = time.perf_counter()

        path = np.array(nx.shortest_path(rm, get_nearest_node((64, 16), rm), get_nearest_node((130, 130), rm)))
        path = end_path([64, 16], [130, 130], path)
        path = cv2.approxPolyDP(path, 8, False)
        path = path.reshape((len(path), 2)).tolist()


        tme = time.perf_counter()
        print(f"Find path: {tme - tms}")
        print(path)

        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for l in rm.edges:
            img = cv2.line(img, l[0], l[1], [0, 0, 255], 1)
        for p in rm:
            img = cv2.circle(img, p, 1, (255, 0, 0), -1)
        for i in range(1, len(path)):
            img = cv2.line(img, path[i-1], path[i], [255, 255, 0], 1)
            img = cv2.circle(img, path[i], 3, (255, 255, 0), -1)

        cv2.imshow("DEBUG", img)
