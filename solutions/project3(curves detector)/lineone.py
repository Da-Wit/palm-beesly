import copy
import solutions.utils as utils
import random


class LineOne:
    def __init__(self, number_of_front_points_to_find_slope):
        self.all_point_list = []
        self.point_list_in_work_area = []
        self.own_slope = -1
        self.number_of_front_points_to_find_slope = number_of_front_points_to_find_slope
        self.changed_after_calculating_slope = True

        # color는 디버깅 용으로 hline별로 쉽게 구별하기 위해 넣은 변수임
        # TODO 실제 개발 완료 시 color 멤버 변수 제거하기
        random_B = random.randint(0, 255)
        random_G = random.randint(0, 255)
        random_R = random.randint(0, 255)

        self.color = [random_B, random_G, random_R]

    # point는 [x, y]꼴임
    def add_point(self, point):
        self.all_point_list.append(point)
        self.point_list_in_work_area.append(point)

    def is_continuable(self, external_point, max_distance=3):
        for idx in range(len(self.point_list_in_work_area)):
            current_point = self.point_list_in_work_area[idx]
            if utils.distance_between(current_point, external_point) < max_distance:
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

            sum_gradient = sum_gradient + \
                           utils.get_slope(front_point, back_point)

        avg_gradient = sum_gradient / half
        self.own_slope = avg_gradient

    def avg_slope_with(self, point):
        sum_gradient = 0
        for i in range(self.number_of_front_points_to_find_slope):
            sum_gradient = sum_gradient + \
                           utils.get_slope(point, self.all_point_list[-i])
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

    def get_min_max_of_x_or_y(self, is_horizontal):
        if is_horizontal:
            index = 0  # index of x
        else:
            index = 1  # index of y

        max_val = max(self.all_point_list, key=lambda point: point[index])[index]
        min_val = min(self.all_point_list, key=lambda point: point[index])[index]

        return min_val, max_val

    def separate(self, filtered, flattening_distance, index):
        copied = copy.deepcopy(filtered)
        separated = []
        temp = []
        # TODO 변수 index, idx, filtered 이름 다시 짓기
        # 변수명이 not specific, not detail하다.

        idx = abs(index - 1)
        for i in copied:
            if len(temp) == 0:
                temp.append(i)
            elif abs(temp[-1][idx] - i[idx]) <= flattening_distance:
                temp.append(i)
            else:
                separated.append(temp)
                temp = [i]
        if len(temp) > 0:
            separated.append(temp)
        return separated

    def flatten_on_one_x_or_y(self, filtered, index):
        idx = abs(index - 1)
        sum_val = 0
        for i in filtered:
            sum_val += i[idx]
        avg = round(sum_val / len(filtered))
        result = [0, 0]
        result[index] = filtered[0][index]
        result[idx] = avg

        return [result]

    def flatten(self, flattening_distance, min_val, max_val, is_horizontal):
        flattened = []

        for i in range(min_val, max_val + 1):
            if is_horizontal:
                index = 0  # index of x
            else:
                index = 1  # index of y

            flattened_on_one_x_or_y = list(filter(lambda point: point[index] == i, self.all_point_list))

            if len(flattened_on_one_x_or_y) == 0:
                continue
            elif len(flattened_on_one_x_or_y) == 1:
                flattened = flattened + flattened_on_one_x_or_y
            else:
                separated = self.separate(flattened_on_one_x_or_y, flattening_distance, index)
                for i in separated:
                    flattened += self.flatten_on_one_x_or_y(i, index)

        self.all_point_list = flattened
