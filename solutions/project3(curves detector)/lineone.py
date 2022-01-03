import solutions.utils as utils
import random

기울기구할때쓸맨앞점들수 = 4


class LineOne:
    def __init__(self):
        self.point_list = []
        self.own_slope = -1
        self.number_of_front_points_to_find_slope = 기울기구할때쓸맨앞점들수

        # color는 디버깅 용으로 hline별로 쉽게 구별하기 위해 넣은 변수임
        # TODO 실제 개발 완료 시 color 멤버 변수 제거하기
        random_B = random.randint(0, 255)
        random_G = random.randint(0, 255)
        random_R = random.randint(0, 255)

        self.color = [random_B, random_G, random_R]

    # point는 [x, y]꼴임
    def add_point(self, point):
        self.point_list.append(point)

    def is_continuable(self, external_point, max_distance=3):
        for idx in range(len(self.point_list)):
            current_point = self.point_list[idx]
            if utils.distance_between(current_point, external_point) < max_distance:
                return True

        return False

    def have_own_slope(self):
        if self.own_slope == -1:
            return False
        return True

    def calculate_own_slope(self):
        half = round(self.number_of_front_points_to_find_slope / 2)
        sum_gradient = 0

        for idx in range(half):
            front_point = self.point_list[idx]
            back_point = self.point_list[idx + half]

            sum_gradient = sum_gradient + utils.get_slope(front_point, back_point)

        avg_gradient = sum_gradient / half
        self.own_slope = avg_gradient

    def avg_slope_with(self, point):
        sum_gradient = 0
        for i in range(self.number_of_front_points_to_find_slope):
            sum_gradient = sum_gradient + utils.get_slope(point, self.point_list[i])
        avg_gradient = sum_gradient / self.number_of_front_points_to_find_slope

        return avg_gradient


def get_indices_of_nearby_lines(nline, point, max_distance):
    index_list = []
    for idx in range(len(nline)):
        is_continuable = nline[idx].is_continuable(point, max_distance)
        if is_continuable:
            index_list.append(idx)
    return index_list


def set_line_info(lines, point, max_distance):
    find = False
    for line in lines:
        if line.is_continuable(point, max_distance) is True:
            find = True
            line.add_point(point)
    if find is False:
        line = LineOne()
        lines.append(line)
        line.add_point(point)

    return find


def append_point(lines, point, max_distance):
    close_lines = get_indices_of_nearby_lines(lines, point, max_distance)
    length_of_close_lines = len(close_lines)

    # 점 주변에 선이 0개일 때
    # 새로운 hline 만들어서 nline에 append
    # 즉, 새로운 선 발견했다고 추정해 새로운 선 추가
    if length_of_close_lines == 0:
        lineOne = LineOne()
        lines.append(lineOne)
        lineOne.add_point(point)

    # 점 주변에 선이 1개일 때
    # 그 1개의 선에 점 추가
    elif length_of_close_lines == 1:
        lineOne = lines[close_lines[0]]
        lineOne.add_point(point)


    # 점 주변에 선이 1개보다 많을 때
    # 기울기로 구함
    else:
        filtered_lines = []

        for i in range(length_of_close_lines):
            hline = lines[close_lines[i]]

            if len(hline.point_list) < 기울기구할때쓸맨앞점들수:
                continue
            elif hline.have_own_slope() is False:
                hline.calculate_own_slope()

            gradient_with_xy = hline.avg_slope_with(point)
            lines_own_gradient = hline.own_slope
            gap = abs(gradient_with_xy - lines_own_gradient)
            filtered_lines.append({"index": close_lines[i], "gap": gap})

        # TODO 랜덤이 아닌 합리적인 방법으로 추가할 선 선택하기
        # 모든 선이 기울기를 구할 수 없을 때
        # 무작위 선 하나에 점을 추가
        if len(filtered_lines) == 0:
            random_index = random.randint(0, length_of_close_lines - 1)
            lines[close_lines[random_index]].add_point(point)


        else:
            min_gap = filtered_lines[0]['gap']
            min_gap_idx = 0
            for i in range(1, len(filtered_lines)):
                if min_gap > filtered_lines[i]['gap']:
                    min_gap = filtered_lines[i]['gap']
                    min_gap_idx = i

            lines[min_gap_idx].add_point(point)
