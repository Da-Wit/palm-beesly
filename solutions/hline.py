import utils
import random
import copy


class Hline:
    def __init__(self):
        self.pointlist = []
        self.color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        return

    def add_line(self, w, h):
        self.pointlist.append([w, h])

    def is_continuable(self, x, y, max_distance=3):
        for i in range(len(self.pointlist)):
            if utils.get_distance(self.pointlist[i], [x, y]) < max_distance:
                return i
        return -1


def set_line_info(nline, x, y, max_distance, for_debugging):
    find = False
    for n in nline:
        if n.is_continuable(x, y, max_distance) > -1:
            find = True
            n.add_line(x, y)
            for_debugging[y][x] = copy.deepcopy(n.color)
    if find is False:
        hl = Hline()
        nline.append(hl)
        hl.add_line(x, y)
        for_debugging[y][x] = copy.deepcopy(hl.color)

    return find
