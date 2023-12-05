from functools import lru_cache
from itertools import product, islice, count
from collections import namedtuple
from math import sqrt, prod
import heapq
import types

vec2 = namedtuple('vec2', ['x', 'y'])
vec3 = namedtuple('vec3', ['x', 'y', 'z'])
vec4 = namedtuple('vec4', ['x', 'y', 'z', 'w'])

def to_vec(t):
    if len(t) == 2:
        return vec2(*t)
    elif len(t) == 3:
        return vec3(*t)
    elif len(t) == 4:
        return vec4(*t)
    return t
    
def add_t(a, b):
    if len(a) == 2:
        return vec2(a[0] + b[0], a[1] + b[1])
    elif len(a) == 3:
        return vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2])
    return to_vec(tuple(sum(v) for v in zip(a, b)))

def sub_t(a, b):
    if len(a) == 2:
        return vec2(a[0] - b[0], a[1] - b[1])
    elif len(a) == 3:
        return vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2])
    return to_vec(tuple(v1 - v2 for v1, v2 in zip(a, b)))

def scale_t(a, b):
    if len(a) == 2:
        return vec2(a[0] * b, a[1] * b)
    elif len(a) == 3:
        return vec3(a[0] * b, a[1] * b, a[2] * b)
    return to_vec(tuple(sum(v) for v in zip(a, b)))

def dot_t(a, b):
    return to_vec(tuple(prod(v) for v in zip(a, b)))

def lt_t(a, b):
    if len(a) == 2:
        return a[0] < b[0] or a[1] < b[1]
    elif len(a) == 3:
        return a[0] < b[0] or a[1] < b[1] or a[2] < b[2]
    return any(v1 < v2 for v1, v2 in zip(a, b))

def gt_t(a, b):
    if len(a) == 2:
        return a[0] > b[0] or a[1] > b[1]
    elif len(a) == 3:
        return a[0] > b[0] or a[1] > b[1] or a[2] > b[2]
    return any(v1 > v2 for v1, v2 in zip(a, b))

def in_dims(addr, dims):
    return all(v >= 0 and v < dims[i] for i, v in enumerate(addr))

def in_grid(addr, grid):
    return addr in grid

def minmax_grid(grid):
    dims = len(next(iter(grid)))
    mi = to_vec(tuple(min(grid, key = lambda v: v[i])[i] for i in range(dims)))
    ma = to_vec(tuple(max(grid, key = lambda v: v[i])[i] for i in range(dims)))
    return mi, ma

def all_addrs(dims):
    return (to_vec(v) for v in product(*(list(range(v) for v in dims))))

def find_addrs(grid, predicate):
    if isinstance(predicate, types.LambdaType) or isinstance(predicate, types.FunctionType):
        return (addr for addr in grid.keys if predicate(grid[addr]))
    return (addr for addr in grid.keys if grid[addr] == predicate)

def gen_grid(dims, initial_value = None):
    addrs = list(all_addrs(dims))
    return dict(zip(addrs, [initial_value] * len(addrs)))

@lru_cache
def _get_offsets(n):
    offsets = []
    for i in range(n):
        offsets.append([0] * i + [1] + [0] * (n - i - 1))
        offsets.append([0] * i + [-1] + [0] * (n - i - 1))
    return offsets

def adj4(addr, dims = None):
    has_dims = isinstance(dims, tuple)
    if not dims:
        n = len(addr)
    else:
        n = dims if not has_dims else len(dims)
    
    addrs = (add_t(addr, offset) for offset in _get_offsets(n))
    return addrs if not has_dims else (a for a in addrs if in_dims(a, dims))

@lru_cache
def _get_offsets_8(n):
    return (v for v in product([-1, 0, 1], repeat=n) if any(v))
    
def adj8(addr, dims = None):
    has_dims = isinstance(dims, tuple)
    if not dims:
        n = len(addr)
    else:
        n = dims if isinstance(dims, int) else len(dims)

    addrs = (add_t(addr, offset) for offset in _get_offsets_8(n))
    return addrs if not has_dims else (a for a in addrs if in_dims(a))

def len_euclid(addr):
    if len(addr) == 2:
        return sqrt(addr[0]**2 + addr[1]**2)
    return sqrt(sum(v**2 for v in addr))

def dist_euclid(a, b):
    return len_euclid(sub_t(b, a))

def len_taxi(addr):
    if len(addr) == 2:
        return abs(addr[0]) + abs(addr[1])
    return sum(abs(v) for v in addr)

def dist_taxi(a, b):
    return len_taxi(sub_t(b, a))

def print_grid(grid, dims, offset = None, width=0, separator=", ", flip_y = False):
    if not offset:
        offset = [0] * len(dims)
    if len(dims) == 2:
        for y in range(dims[1]) if not flip_y else reversed(range(dims[1])):
            print(separator.join((str(grid[(x + offset[0], y + offset[1])]) if (x + offset[0], y + offset[1]) in grid else "").rjust(width) for x in range(dims[0])))
    elif len(dims) == 3:
        for z in range(dims[2]):
            print(f"layer {z}")
            for y in range(dims[1]) if not flip_y else reversed(range(dims[1])):
                print("    " + separator.join((str(grid[(x + offset[0], y + offset[1], z + offset[2])]) if (x + offset[0], y + offset[1], z + offset[2]) in grid else "").rjust(width) for x in range(dims[0])))
    else:
        print(f"can't print grid of len {len(dims)}")


def auto_print_grid_2d(grid):
    mi, ma = minmax_grid(grid)

    for j in range(mi.y, ma.y + 1):
        for i in range(mi.x, ma.x + 1):
            print(grid[vec2(i, j)] if vec2(i, j) in grid else ".", end="")
        print()
    print()

def reconstruct_path(start, node, previous_nodes):
    path = [node]
    while node != start:
        node = previous_nodes[node]
        path.append(node)
    path.append(start)
    return reversed(path)

def djikstra(start, graph, neighbours = None, cost = None, max_value = None):
    if not neighbours:
        neighbours = lambda n: (a for a in adj4(n) if a in graph)

    if not cost:
        cost = lambda _, __: 1

    if not max_value:
        max_value = float("inf")

    shortest_paths = {start: 0}

    shortest_path = lambda n: shortest_paths[n] if n in shortest_paths else max_value

    previous_nodes = {}
    visited = set()
    
    for n in neighbours(start):
        shortest_paths[n] = cost(start, n)
        previous_nodes[n] = start

    frontier = set(neighbours(start))
    while frontier:
        min_node = min(frontier, key=shortest_path)
        frontier.remove(min_node)
        visited.add(min_node)
        
        for neighbour in neighbours(min_node):
            tentative = shortest_path(min_node) + cost(min_node, neighbour)
            if tentative < shortest_path(neighbour):
                shortest_paths[neighbour] = tentative
                previous_nodes[neighbour] = min_node
            if neighbour not in visited:
                frontier.add(neighbour)

    return (previous_nodes, shortest_paths)
            
def astar(start, goal, graph, neighbours = None, cost = None, heuristic = None, max_value = None):
    if not neighbours:
        neighbours = lambda n: (a for a in adj4(n) if a in graph)

    if not heuristic:
        g = goal
        heuristic = lambda n: dist_taxi(n, g)

    if not cost:
        cost = lambda _, __: 1

    if not max_value:
        max_value = float("inf")

    if not (isinstance(goal, types.LambdaType) or isinstance(goal, types.FunctionType)):
        g = goal
        goal = lambda n: n == g

    previous_nodes = {}  

    # TODO probably pre-populate previous_nodes/shortest_paths for neighbours(start)

    shortest_paths = {start: 0}
    shortest_path = lambda n: shortest_paths[n] if n in shortest_paths else max_value

    frontier = [(0, start)]
    heapq.heapify(frontier)
    frontier_set = { start }

    while frontier:
        _, node = heapq.heappop(frontier)
        frontier_set.remove(node)

        #print(node)
        #print(goal(node))
        if goal(node):
            return reconstruct_path(start, node, previous_nodes)
        
        for neighbour in neighbours(node):
            tentative = shortest_path(node) + cost(node, neighbour)
            if tentative < shortest_path(neighbour):
                previous_nodes[neighbour] = node
                shortest_paths[neighbour] = tentative
                if neighbour not in frontier_set:
                    heapq.heappush(frontier, (tentative + heuristic(neighbour), neighbour))
                    frontier_set.add(neighbour)

    return None

def window(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def window_indices(n):
    return (list(range(i, i + n)) for i in count())

def distinct(seq):
    return list(set(seq))

def transpose(seq):
    return list(zip(*seq))

def chunk(seq, size):
    seq = iter(seq)
    return iter(lambda: list(islice(seq, size)), [])


def aaline(pa, pb):
    pa, pb = to_vec(pa), to_vec(pb)
    length = dist_taxi(pa, pb)
    dir = sub_t(pb, pa)
    dir = to_vec(tuple(v / abs(v) if v != 0 else 0 for v in dir))
    yield pa
    for _ in range(length):
        pa = add_t(pa, dir)
        yield to_vec(tuple(int(v) for v in pa))