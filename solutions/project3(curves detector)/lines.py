from lineone import LineOne
import numpy as np
import random
import copy
import cv2
import os

class Lines:
    def __init__(self):
        self.line_list = []
        self.number_of_front_points_to_find_slope = 4

    def add_line(self, line):
        self.line_list.append(line)

    def renew_work_area(self, point, max_distance):
        for lineOne in self.line_list:
            lineOne.renew_work_area(point, max_distance)

    def get_index_list_of_close_lines(self, point, max_distance):
        index_list = []
        for idx in range(len(self.line_list)):
            is_continuable = self.line_list[idx].is_continuable(point, max_distance)
            if is_continuable:
                index_list.append(idx)
        return index_list

    def set_line_info(self, point, max_distance):
        find = False
        for line in self.line_list:
            if line.is_continuable(point, max_distance) is True:
                find = True
                line.add_point(point)
        if find is False:
            line = LineOne(self.number_of_front_points_to_find_slope)
            self.line_list.append(line)
            line.add_point(point)

        return find

    def handle_point(self, point, max_distance):
        list_of_index_of_close_lines = self.get_index_list_of_close_lines(point, max_distance)
        number_of_close_lines = len(list_of_index_of_close_lines)

        # 점 주변에 선이 0개일 때
        # 새로운 hline 만들어서 nline에 append
        # 즉, 새로운 선 발견했다고 추정해 새로운 선 추가
        if number_of_close_lines == 0:
            lineOne = LineOne(self.number_of_front_points_to_find_slope)
            lineOne.add_point(point)
            self.add_line(lineOne)

        # 점 주변에 선이 1개일 때
        # 그 1개의 선에 점 추가
        elif number_of_close_lines == 1:
            index = list_of_index_of_close_lines[0]
            lineOne = self.line_list[index]
            lineOne.add_point(point)

        # 점 주변에 선이 1개보다 많을 때
        # 기울기로 구함
        else:
            filtered_lines = []

            for index_of_line in list_of_index_of_close_lines:
                lineOne = self.line_list[index_of_line]

                if len(lineOne.all_point_list) < self.number_of_front_points_to_find_slope:
                    continue
                elif lineOne.have_own_slope() is False or lineOne.changed_after_calculating_slope is True:
                    lineOne.calculate_own_slope()
                    lineOne.changed_after_calculating_slope = False

                slope_related_to_xy = lineOne.avg_slope_with(point)
                line_own_slope = lineOne.own_slope
                gap = abs(slope_related_to_xy - line_own_slope)
                filtered_lines.append({"index": index_of_line, "gap": gap})

            # TODO 랜덤이 아닌 합리적인 방법으로 추가할 선 선택하기
            # 모든 선이 기울기를 구할 수 없을 때
            # 무작위 선 하나에 점을 추가
            if len(filtered_lines) == 0:
                random_index = random.randint(0, number_of_close_lines - 1)
                self.line_list[list_of_index_of_close_lines[random_index]].add_point(point)

            else:
                min_gap = filtered_lines[0]['gap']
                min_gap_idx = filtered_lines[0]['index']
                for index_of_line in range(1, len(filtered_lines)):
                    if min_gap > filtered_lines[index_of_line]['gap']:
                        min_gap = filtered_lines[index_of_line]['gap']
                        min_gap_idx = filtered_lines[index_of_line]['index']

                self.line_list[min_gap_idx].add_point(point)

    def filter_by_line_length(self, min_length):
        line_list = copy.deepcopy(self.line_list)
        for lineOne in line_list:
            if len(lineOne.all_point_list) < min_length:
                line_list.remove(lineOne)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.
        self.line_list = line_list


    def temp_fucntion(self, img_param, image_name):
        directory_path = "C:/Users/think/workspace/palm-beesly/test_img"
        copied_img = copy.deepcopy(img_param)
        height, width = copied_img.shape[:2]
        for_showing = np.zeros((height, width, 1), dtype=np.uint8)
        count = 0

        for lineOne in self.line_list:
            for point in lineOne.all_point_list:
                x,y = point
                for_showing[y][x] = 255
            cv2.imwrite(os.path.join(directory_path, f"{image_name}{count}.png"), for_showing)
            for_showing = for_showing * 0
            count += 1


    def visualize_lines(self, img_param, color=False):
        copied_img = copy.deepcopy(img_param)
        height, width = copied_img.shape[:2]
        if color:
            copied_img = cv2.cvtColor(copied_img, cv2.COLOR_GRAY2BGR)
            temp = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            temp = np.zeros((height, width, 1), dtype=np.uint8)

        for lineOne in self.line_list:
            temp = temp * 0
            for point in lineOne.all_point_list:

                x = point[0]
                y = point[1]

                if color:
                    temp[y][x] = lineOne.color
                else:
                    temp[y][x] = 255

            copied_img = copied_img + temp

            # cv2.imshow("img_on_progress", utils.resize(for_showing, width=600))
            # k = cv2.waitKey(0)
            # if k == 27:  # Esc key to stop
            #     cv2.destroyAllWindows()
            #     exit(0)

        return copied_img + temp

    def sort(self):
        line_list = copy.deepcopy(self.line_list)

        def criteria(lineOne):
            return len(lineOne.all_point_list)

        line_list.sort(reverse=True, key=criteria)
        self.line_list = line_list

    def leave_long_lines(self, number_of_lines_to_leave=10):
        self.sort()

        if number_of_lines_to_leave > len(self.line_list):
            print("number_of_lines_to_leave can't be larger than length of line_list.")
            exit(1)
        self.line_list = self.line_list[:number_of_lines_to_leave]
