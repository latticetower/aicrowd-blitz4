import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from skimage.morphology import convex_hull_image
from skimage.measure import label as skimage_label


@dataclass
class Puzzle:
    id: int
    path: str
    def iterate(self):
        for file in os.listdir(self.path):
            yield os.path.join(self.path, file)
    def read(self):
        return [(piece, cv2.imread(piece, cv2.IMREAD_UNCHANGED))
        for piece in self.iterate()]


@dataclass
class ExtractionResult:
    img:str
    contour: str
    qhull : str
    intersection : str
    colored_intersection: str
    other: str
    inner: str
    outer: str

def filter_by_size(colored_intersection, min_cutoff=10):
    colors = np.unique(colored_intersection)
    sel_colors = np.zeros_like(colored_intersection)
    for color in colors:
        if color == 0:
            continue
        size = (colored_intersection == color).sum()
        if size > min_cutoff:
            sel_colors += color*(colored_intersection == color)
        #print(color, (colored_intersection == color).sum())
    return  sel_colors

def check_horizontal(contour, pos, cy, cx):
    h, w = contour.shape
    ids = (pos[:, 0] > cy-1) & (pos[:, 0] < cy+1)
    sel_w = pos[ids, 1]
    min_w = sel_w.min()
    max_w = sel_w.max()
    if min_w <= 1:
        return False
    if max_w >= w-2:
        return False
    #print(pos[ids])
    #p1 = contour[min_h - 2, int(cx)]
    #p2 = contour[max_h + 2, int(cx)]
    points = np.argwhere(contour)
    p_ids = (points[:, 0] > cy-1) & (points[:, 0] < cy+1)
    p_min = np.any(points[p_ids, 1]< min_w)
    p_max = np.any(points[p_ids, 1] > max_w)
    return p_min and p_max

def check_vertical(contour, pos, cy, cx):
    h, w = contour.shape
    ids = (pos[:, 1] > cx-1) & (pos[:, 1] < cx+1)
    sel_h = pos[ids, 0]
    min_h = sel_h.min()
    max_h = sel_h.max()
    if min_h <= 1:
        return False
    if max_h >= h-2:
        return False
    #print(pos[ids])
    #p1 = contour[min_h - 2, int(cx)]
    #p2 = contour[max_h + 2, int(cx)]
    points = np.argwhere(contour)
    p_ids = (points[:, 1] > cx-1) & (points[:, 1] < cx+1)
    p_min = np.any(points[p_ids, 0]< min_h)
    p_max = np.any(points[p_ids, 0] > max_h)
    return p_min and p_max
    #print(p1, p2, p1==p2, min_h - 1, int(cx))

def select_h(pos, contour):
    xpos = np.unique(pos[:, 1])
    #print(len(xpos))
    points = np.argwhere(contour)
    new_x = []
    for x in xpos:
        ys = pos[pos[:, 1] == x, 0]
        p_ids = points[:, 1] == x
        c1 = np.any(points[p_ids, 0] > ys.mean())
        c2 = np.any(points[p_ids, 0] < ys.mean())
        holes = [i for i in np.arange(ys.min(), ys.max())
        if i not in ys]
        c3 = len(holes) <= 0
        if c1 and c2 and c3:
            new_x.append(x)
        #else:
        #    print(x)
    #print(new_x)
    if len(new_x) == 0:
        return []
    collected = np.stack([pos[:, 1] == x for x in new_x]).any(0)
    return pos[collected]

def select_w(pos, contour):
    ypos = np.unique(pos[:, 0])
    #print(len(ypos))
    points = np.argwhere(contour)
    new_y = []
    for y in ypos:
        xs = pos[pos[:, 0] == y, 1]
        p_ids = points[:, 0] == y
        c1 = np.any(points[p_ids, 1] > xs.mean())
        c2 = np.any(points[p_ids, 1] < xs.mean())
        holes = [i for i in np.arange(xs.min(), xs.max())
        if i not in xs]
        c3 = len(holes) <= 0
        if c1 and c2 and c3:
            new_y.append(y)
        #else:
        #    print(x)
    #print("new_y", new_y)
    if len(new_y) == 0:
        return []
    collected = np.stack([pos[:, 0] == y for y in new_y]).any(0)
    return pos[collected]

def extract_inner(colored_intersection, contour):
    colors = np.unique(colored_intersection)
    h, w = contour.shape
    inner_h = np.zeros_like(colored_intersection)
    inner_w = np.zeros_like(colored_intersection)
    outer = np.zeros_like(colored_intersection)

    for color in colors:
        if color == 0:
            continue
        ctype = None
        pos = np.argwhere(colored_intersection==color)
        cy, cx = pos.mean(0)
        if check_horizontal(contour, pos, cy, cx):
            pos_ = select_w(pos, contour)
            #pos_ = pos
            #print(pos.shape)
            if len(pos_) > 0:
                inner_w[pos_[:, 0], pos_[:, 1]] = color
            else:
                print("invalid pos", pos.shape)
            ctype = 'horizontal'
        if check_vertical(contour, pos, cy, cx):
            if ctype is not None:
                raise ("found ctype with both vertical and horizontal")
            pos_ = select_h(pos, contour)
            #pos_ = pos
            if len(pos) > 0:
                inner_h[pos_[:, 0], pos_[:, 1]] = color
            else:
                print("invalid vertical", pos.shape)
            ctype = 'vertical'
        if ctype is None:
            outer[pos[:, 0], pos[:, 1]] = color
    return inner_h, inner_w, outer #colored_intersection, colored_intersection

def expand_inner(inner_h, inner_w):
    h, w = inner_h.shape
    down = np.zeros_like(inner_h)
    up = np.zeros_like(inner_h)
    colors = np.unique(inner_h)
    for color in colors:
        if color == 0:
            continue
        pos = np.argwhere(inner_h == color)
        if np.all(pos[:, 0] > h/2):
            down[pos[:, 0], pos[:, 1]] = color
        else:
            up[pos[:, 0], pos[:, 1]] = color

    right = np.zeros_like(inner_w)
    left = np.zeros_like(inner_w)
    colors = np.unique(inner_w)
    for color in colors:
        if color == 0:
            continue
        pos = np.argwhere(inner_w == color)
        if np.all(pos[:, 1] > w/2):
            right[pos[:, 0], pos[:, 1]] = color
        else:
            left[pos[:, 0], pos[:, 1]] = color
    if down.std() == 0:
        down = None
    if up.std() == 0:
        up = None
    if right.std() == 0:
        right = None
    if left.std() == 0:
        left = None
    return [down, up, right, left]

def extract_outer(outer, contour):
    h, w = contour.shape
    colors = np.unique(outer)
    if len(colors) <= 1:
        return [None, None, None, None]
    cpos = np.argwhere(contour)
    cy, cx = cpos.mean(0)
    outer_down = np.zeros_like(outer)
    outer_up = np.zeros_like(outer)
    outer_right = np.zeros_like(outer)
    outer_left = np.zeros_like(outer)
    for color in colors:
        if color == 0:
            continue
        pos = np.argwhere(outer == color)
        pos_h = pos[:, 0]
        pos_w = pos[:, 1]
        min_h = pos_h.min()
        max_h = pos_h.max()
        min_w = pos_w.min()
        max_w = pos_w.max()
        down = (cpos[:, 0] >= min_h).sum()
        up = (cpos[:, 0] <= max_h).sum()
        right = (cpos[:, 1] >= min_w).sum()
        left = (cpos[:, 1] <= max_w).sum()
        mpos = np.min([down, up, right, left])
        assert mpos > 0
        if mpos == down:
            ids = cpos[:, 0] >= min_h - 5
            py = cpos[ids, 0]
            px = cpos[ids, 1]
            outer_down[py, px] = color
        if mpos == up:
            ids = cpos[:, 0] <= max_h + 5
            py = cpos[ids, 0]
            px = cpos[ids, 1]
            outer_up[py, px] = color
        if mpos == right:
            ids = cpos[:, 1] >= min_w - 5
            py = cpos[ids, 0]
            px = cpos[ids, 1]
            outer_right[py, px] = color
        if mpos == left:
            ids = cpos[:, 1] <= max_w + 5
            py = cpos[ids, 0]
            px = cpos[ids, 1]
            outer_left[py, px] = color
    if outer_down.std() == 0:
        outer_down = None
    if outer_up.std() == 0:
        outer_up = None
    if outer_left.std() == 0:
        outer_left = None
    if outer_right.std() == 0:
        outer_right = None
    return [outer_down, outer_up, outer_right, outer_left]

def extract_parts(img):
    contour = img[:, :, 3]
    contour = contour > 0
    qhull = convex_hull_image(contour)
    intersection = qhull.astype(int) - contour
    colored_intersection = skimage_label(intersection)
    sel_colors = filter_by_size(colored_intersection)
    colored_intersection = sel_colors
    inner_h, inner_w, outer = extract_inner(colored_intersection, contour)
    outer_expanded = extract_outer(outer, contour)
    inner_expanded = expand_inner(inner_w, inner_h)
    return ExtractionResult(
        img,
        contour, qhull, intersection, colored_intersection,
        other=[inner_h, inner_w, outer],
        inner=inner_expanded,
        outer=outer_expanded)


def crop(x, type='both'):
    coords = np.argwhere(x > 0)
    hmin, wmin = coords.min(0)
    hmax, wmax = coords.max(0)+1
    if type == 'both':
        return x[hmin:hmax, wmin:wmax], hmin, wmin
    return x[hmin:hmax, :], hmin, 0


def crop_top(x):
    coords = np.argwhere(x > 0)
    hmin, wmin = coords.min(0)
    return x[hmin:], hmin

def align(x1, x2):
    h1, w1 = x1.shape
    h2, w2 = x2.shape
    h = min(h1, h2)
    cy1, cx1 = np.argwhere(x1[:h//2] > 0).mean(0).astype(int)
    cy2, cx2  = np.argwhere(x2[:h//2] > 0).mean(0).astype(int)
    # were are interested in cx1 and cx2 alignment
    left = min(cx1, cx2)
    w = min(w1 - cx1, w2 - cx2)
    #print(cy1, cx1, cy2, cx2)
    return x1[:h, cx1-left:cx1+ w], x2[:h, cx2-left:cx2+ w]
    pass

def match(x1, x2, show=False, align_type='best'):
    crop_type = 'both' if align_type == 'best' else 'w'
    x1, _, _ = crop(x1, type=crop_type)
    if show:
        plt.imshow(x1)
        plt.title("x1")
        plt.show()
    x2, _, _ = crop(x2, type=crop_type)
    if show:
        plt.imshow(x2)
        plt.title("x2")
        plt.show()
    h1, w1 = x1.shape
    h2, w2 = x2.shape
    if align_type == 'best':
        x1, x2 = align(x1, x2)
    elif w1 != w2:
        #h = min(h1, h2)
        w = min(w1, w2)
        if align_type == "left":
            x1 = x1[:, :w]
            x2 = x2[:, :w]
        else:
            x1 = x1[:, -w:]
            x2 = x2[:, -w:]
    if h1 != h2:
        h = min(h1, h2)
        x1 = x1[:h]
        x2 = x2[:h]

    if show:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(x1)
        ax2.imshow(x2)
        plt.show()

    intersection = (x1 > 0) * (x2 > 0)
    union = ((x1 > 0) + (x2 > 0)) > 0
    diff = (x1 > 0) != (x2 > 0) # (x1 > 0) != union
    if show:
        plt.imshow(diff)
        plt.show()
        plt.imshow(intersection)
        plt.show()
        plt.imshow(union)
        plt.show()
        plt.imshow(union * (~intersection))
        plt.show()
    return diff.sum()



def read_puzzle(puzzle, show=False):
    puzzle_data = dict()
    for name, img in puzzle.read():
        contour = img[:, :, 3]
        result = extract_parts(img)
        qhull = result.qhull
        intersection = result.intersection
        puzzle_data[name] = result
        if show:
            fig, axes = plt.subplots(ncols=14, nrows=1, figsize=(14, 1))
            ax0, ax1, ax2 = axes[:3]
            ax0.imshow(contour)
            ax1.imshow(qhull)
            ax2.imshow(result.colored_intersection)
            for ax, img in zip(axes[3:], result.other+result.inner+result.outer):
                if img is None:
                    continue
                ax.imshow(img)
            plt.show()
    return puzzle_data


inner_transform = {
    'down': lambda x: x,
    'up': lambda x: np.rot90(x, 2),
    'right': lambda x: np.rot90(x, 3),
    'left': lambda x: np.rot90(x, 1)
}
outer_transform = {
    'down': lambda x: np.rot90(x, 2),
    'up': lambda x: x,
    'right': lambda x: np.rot90(x, 1),
    'left': lambda x: np.rot90(x, 3)
}


def compare(res, inner_transformed, outer_transformed):
    labels = ['down', 'up', 'right', 'left']
    for i, label in zip(res.inner, labels):
        if i is None:
            continue
        i = inner_transform[label](i)
        for o, path, l2 in outer_transformed:
            m = match(i, o)
            if m > 20:
                continue
            print(label, path, l2, m)
            match(i, o, show=True)
    for o, label in zip(res.outer, labels):
        if o is None:
            continue
        o = outer_transform[label](o)
        for i, path, l2 in inner_transformed:
            m = match(i, o)
            if m > 20:
                continue
            print(label, path, l2, m)
            match(i, o, show=True)
        #plt.imshow(o)
        #plt.show()
    pass


class Collection:
    def __init__(self, height, width, puzzle_data):
        self.height = height
        self.width = width
        self.field = [
            [
                set() for _ in range(width)]
            for line in range(height)
        ]
        self.puzzle_data = puzzle_data

    def fill(self, y, x, path):
        self.field[y][x] = path
    def draw(self):
        fig, axes = plt.subplots(ncols=self.width, nrows=self.height, figsize=(self.height*2, self.width*2))
        for i in range(self.height):
            for j in range(self.width):
                cell = self.field[i][j]
                #print(cell)
                ax = axes[i][j]
                if isinstance(cell, str):
                    img = self.puzzle_data[cell].img
                    ax.imshow(img)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        #for line, axes2 in zip(self.field, axes):
        #    for cell, ax in zip(line, axes2):
        #        if isinstance(cell, string) and len(cell)>6:
        #            img = self.puzzle_data[cell].img
        #            ax.imshow(img)
        plt.show()
