class Hline:
    def __init__(self):
        self.pointlist = []
        return

    def add_line(self, w, h):
        self.pointlist.append([w, h])

    def is_continuable(self, w, h, max_distance=3):
        if abs(self.pointlist[-1][1] - h) < max_distance \
                and abs(self.pointlist[-1][0] - w) < max_distance:
            return True
        return False


def set_line_info(nline, w, h):
    find = False
    for n in nline:
        if n.is_continuable(w, h) is True:
            find = True
            n.add_line(w, h)
    if find is False:
        hl = Hline()
        nline.append(hl)
        hl.add_line(w, h)
    return find
