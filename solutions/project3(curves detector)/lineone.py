import solutions.utils as utils
import random


class LineOne:
    def __init__(self, number_of_front_points_to_find_slope):
        self.point_list = []
        self.own_slope = -1
        self.number_of_front_points_to_find_slope = number_of_front_points_to_find_slope

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
