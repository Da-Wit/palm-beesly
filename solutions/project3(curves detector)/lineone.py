from solutions.utils import get_slope
from solutions.utils import distance_between
import random


class LineOne:
    def __init__(self, number_of_front_points_to_find_slope, unique_num):
        self.all_point_list = []
        self.point_list_in_work_area = []
        self.own_slope = -1
        self.number_of_front_points_to_find_slope = number_of_front_points_to_find_slope
        self.changed_after_calculating_slope = True
        self.unique_num = unique_num
        # color는 디버깅 용으로 hline별로 쉽게 구별하기 위해 넣은 변수임
        # TODO 실제 개발 완료 시 color 멤버 변수 제거하기
        random_B = random.randint(1, 255)
        random_G = random.randint(1, 255)
        random_R = random.randint(1, 255)

        self.color = [random_B, random_G, random_R]

    # point는 [x, y]꼴임
    def add_point(self, point):
        self.all_point_list.append(point)
        self.point_list_in_work_area.append(point)

    def set_unique_num_to(self, unique_num):
        self.unique_num = unique_num
        return True

    def is_continuable(self, external_point, max_distance=3):
        for idx in range(len(self.point_list_in_work_area)):
            current_point = self.point_list_in_work_area[idx]
            if distance_between(current_point, external_point) < max_distance:
                return True

        return False

    def have_own_slope(self):
        if self.own_slope == -1:
            return False
        return True

    def calculate_own_slope(self):
        point_list = self.all_point_list[-self.number_of_front_points_to_find_slope:]

        half = round(self.number_of_front_points_to_find_slope / 2)
        sum_gradient = 0

        for idx in range(half):
            front_point = point_list[idx]
            back_point = point_list[idx + half]

            sum_gradient = sum_gradient + get_slope(front_point, back_point)

        avg_gradient = sum_gradient / half
        self.own_slope = avg_gradient

    def avg_slope_with(self, point):
        sum_gradient = 0
        for i in range(self.number_of_front_points_to_find_slope):
            sum_gradient = sum_gradient + get_slope(point, self.all_point_list[-i])
        avg_gradient = sum_gradient / self.number_of_front_points_to_find_slope

        return avg_gradient

    def renew_work_area(self, external_point, max_distance, is_horizontal):
        list_len = len(self.point_list_in_work_area)
        number_of_deleted_point = 0
        coordinate_idx = 0 if is_horizontal else 1

        for idx in range(list_len):
            real_idx = idx - number_of_deleted_point
            current_point = self.point_list_in_work_area[real_idx]
            gap = abs(current_point[coordinate_idx] - external_point[coordinate_idx])

            if gap >= max_distance:
                del self.point_list_in_work_area[real_idx]
                number_of_deleted_point += 1
