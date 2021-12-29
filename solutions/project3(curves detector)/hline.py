import solutions.utils as utils
import random


class Hline:
    def __init__(self):
        self.pointlist = []
        # color는 디버깅 용으로 hline별로 쉽게 구별하기 위해 넣은 변수
        # TODO 실제 개발 완료 시 color 멤버 변수 제거하기
        self.color = [random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)]
        self.gradient = -1
        return

    def add_point(self, x, y):
        self.pointlist.append([x, y])

    def is_continuable(self, x, y, max_distance=3):
        for i in range(len(self.pointlist)):
            if utils.get_distance(self.pointlist[i], [x, y]) < max_distance:
                return i
        return -1

    def avg_gradient_with(self, x, y):
        이것보단커야지_기울기_구함 = 10

        if len(self.pointlist) < 이것보단커야지_기울기_구함:
            return -1
        elif self.gradient == -1:
            기울기구할때쓸맨앞점들수 = 10
            sum_gradient = 0

            for i in range(round(기울기구할때쓸맨앞점들수 / 2)):
                point1 = self.pointlist[i]
                point2 = self.pointlist[i + 5]

                sum_gradient = sum_gradient + utils.get_gradient(point1, point2)

            avg_gradient = sum_gradient / round(기울기구할때쓸맨앞점들수 / 2)
            self.gradient = avg_gradient

        coord1 = [x, y]
        sum_gradient = 0
        for i in range(이것보단커야지_기울기_구함):
            sum_gradient = sum_gradient + utils.get_gradient(coord1, self.pointlist[i])
        avg_gradient = sum_gradient / 이것보단커야지_기울기_구함

        return avg_gradient


def get_indices_of_nearby_lines(nline, x, y, max_distance):
    index_list = []
    for i in range(len(nline)):
        if nline[i].is_continuable(x, y, max_distance) > -1:
            index_list.append(i)
    return index_list


def set_line_info(nline, x, y, max_distance):
    find = False
    for n in nline:
        if n.is_continuable(x, y, max_distance) > -1:
            find = True
            n.add_point(x, y)
    if find is False:
        hl = Hline()
        nline.append(hl)
        hl.add_point(x, y)

    return find


def append_point(nline, x, y, max_distance):
    indices_of_nearby_lines = get_indices_of_nearby_lines(nline, x, y, max_distance)
    number_of_nearby_lines = len(indices_of_nearby_lines)

    # 점 주변에 선이 0개일 때
    # 새로운 hline 만들어서 nline에 append
    # 즉, 새로운 선 발견했다고 추정해 새로운 선 추가
    if number_of_nearby_lines == 0:
        hl = Hline()
        nline.append(hl)
        hl.add_point(x, y)

    # 점 주변에 선이 1개일 때
    # 그 1개의 선에 점 추가
    elif number_of_nearby_lines == 1:
        target = nline[indices_of_nearby_lines[0]]
        target.add_point(x, y)

    # 점 주변에 선이 1개보다 많을 때
    # 기울기로 구함
    else:
        filtered_lines = []
        for i in range(number_of_nearby_lines):
            hline = nline[indices_of_nearby_lines[i]]
            gradient_with_xy = hline.avg_gradient_with(x, y)
            lines_own_gradient = hline.gradient

            # gradient_with_xy이 -1이라는 의미는 hline의 길이가
            # 10보다 작아서 기울기를 구할 만큼 길지 않다는 의미
            # 따라서 아래 if문은 기울기를 구할 수 있는지 여부를 체크하고
            # 구할 수 있을 때만 기울기를 구하는 코드임
            if gradient_with_xy != -1:
                gap = abs(gradient_with_xy - lines_own_gradient)
                filtered_lines.append({"index": indices_of_nearby_lines[i], "gap": gap})

        # TODO 랜덤이 아닌 합리적인 방법으로 추가할 선 선택하기
        # 모든 선이 기울기를 구할 수 없을 때
        # 무작위 선 하나에 점을 추가
        if len(filtered_lines) == 0:
            random_index = random.randint(0, number_of_nearby_lines - 1)
            nline[indices_of_nearby_lines[random_index]].add_point(x, y)

        else:
            min_gap = filtered_lines[0]['gap']
            min_gap_idx = 0
            for i in range(1, len(filtered_lines)):
                if min_gap > filtered_lines[i]['gap']:
                    min_gap = filtered_lines[i]['gap']
                    min_gap_idx = i

            nline[min_gap_idx].add_point(x, y)
