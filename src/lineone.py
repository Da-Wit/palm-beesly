# 이 파일은 선 검출 과정에서 검출된 하나의 선을 LineOne이라는 클래스로 사용할 수 있도록 한다.
# 선 검출이 끝나면 LineOne 하나는 검출된 하나의 선을 의미하여, 하나의 선에 포함된 점들의 좌표를
# all_point_list에 저장하고 있다.
# 원리를 이해하면 선의 순서는 장담할 수 없다는 걸 알 수 있다.

from utils import slope_between
from utils import distance_between
import random


# 기울기와 관련된
# number_of_front_points_to_find_slope,
# changed_after_calculating_slope,
# calculate_own_slope,
# avg_slope_with,
# 는 현재 사용되지 않고 있음
# 이 변수 및 함수들은 옛날 옛적에 선 검출 과정에서,
# 그러니까 lines.handle_point에서
# number_of_close_lines가 2 이상일 때
# 현재처럼 모든 선을 병합하지 않고,
# 모든 선들에 검출된 점을 넣어보고 기존의 기울기와 차이가
# 가장 적은 선에 점을 추가하는 방식이었을 때 사용하였다.
# 레거시긴 한데 재사용 될 가능성이 있어 놔두었다.
# 만약 쓸거면 calculate_own_slope의 알고리즘을 개선하고 쓰는 걸 추천한다.
# 그리고 위에 소개한 것처럼 2개 이상의 선이 발견됐을 때
# 어느 선에 추가할 지를 결정해야 한다면 기울기를 기준으로 할 경우
# 너무 차이가 심하다.
# 그래서 각도를 이용하는 걸 추천한다.

class LineOne:
    def __init__(self, number_of_front_points_to_find_slope, unique_num):
        self.all_point_list = []
        self.point_list_in_work_area = []
        self.own_slope = -1
        self.number_of_front_points_to_find_slope = number_of_front_points_to_find_slope
        self.changed_after_calculating_slope = True
        self.unique_num = unique_num  # 기본적으론 LineOne마다 값이 다름.

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

    # 선 검출 과정 중, 점이 LineOne에 연결할 수 있을 정도로 거리가 가까운 지 확인하는 함수
    def is_continuable(self, external_point, max_distance=3):
        for idx in range(len(self.point_list_in_work_area)):
            current_point = self.point_list_in_work_area[idx]
            if distance_between(current_point, external_point) < max_distance:
                return True

        return False

    # 레거시
    def have_own_slope(self):
        if self.own_slope == -1:
            return False
        return True

    # 레거시
    def calculate_own_slope(self):
        point_list = self.all_point_list[-self.number_of_front_points_to_find_slope:]

        half = round(self.number_of_front_points_to_find_slope / 2)
        sum_gradient = 0

        for idx in range(half):
            front_point = point_list[idx]
            back_point = point_list[idx + half]

            sum_gradient = sum_gradient + slope_between(front_point, back_point)

        avg_gradient = sum_gradient / half
        self.own_slope = avg_gradient

    # 레거시
    def avg_slope_with(self, point):
        sum_gradient = 0
        for i in range(self.number_of_front_points_to_find_slope):
            sum_gradient = sum_gradient + slope_between(point, self.all_point_list[-i])
        avg_gradient = sum_gradient / self.number_of_front_points_to_find_slope

        return avg_gradient

    # 선 검출은 이중 for문이고, 점점 한 방향 시작에서 한 방향 끝으로 간다.
    # 그래서 그 한 방향의 시작에 가까우면서 max_distance보다 점에서 먼 점들을 연산할 필요는 없기에
    # 이 함수는 그러한 점들을 작업 영역(point_list_in_work_area)에서 제거한다.
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
