from lineone import LineOne
import numpy as np
import copy
import cv2

from solutions import utils


class Lines:
    def __init__(self):
        self.line_list = []
        self.number_of_front_points_to_find_slope = 4

    def add_line(self, line):
        self.line_list.append(line)

    def renew_work_area(self, point, max_distance, is_horizontal):
        for lineOne in self.line_list:
            lineOne.renew_work_area(point, max_distance, is_horizontal)

    def get_index_list_of_close_lines(self, point, max_distance):
        index_list = []
        for idx in range(len(self.line_list)):
            is_continuable = self.line_list[idx].is_continuable(point, max_distance)
            if is_continuable:
                index_list.append(idx)
        return index_list

    # TODO change function name
    def combine_only_params(self, indices, new_lineOne):
        for idx in indices:
            lineOne = self.line_list[idx]
            point_list = copy.deepcopy(lineOne.all_point_list)
            new_lineOne.all_point_list += point_list

        for index in range(len(indices) - 1, -1, -1):
            del self.line_list[indices[index]]

        self.add_line(new_lineOne)

    def handle_point(self, point, max_distance, unique_num):
        list_of_index_of_close_lines = self.get_index_list_of_close_lines(point, max_distance)
        number_of_close_lines = len(list_of_index_of_close_lines)

        # 점 주변에 선이 0개일 때
        # 새로운 hline 만들어서 nline에 append
        # 즉, 새로운 선 발견했다고 추정해 새로운 선 추가
        if number_of_close_lines == 0:
            lineOne = LineOne(self.number_of_front_points_to_find_slope, unique_num)
            lineOne.add_point(point)
            self.add_line(lineOne)

        # 점 주변에 선이 1개일 때
        # 그 1개의 선에 점 추가
        elif number_of_close_lines == 1:
            index = list_of_index_of_close_lines[0]
            lineOne = self.line_list[index]
            lineOne.add_point(point)

        # 점 주변에 선이 1개보다 많을 때
        else:
            new_lineOne = LineOne(self.number_of_front_points_to_find_slope, unique_num)
            self.combine_only_params(list_of_index_of_close_lines, new_lineOne)

    def filter_by_line_length(self, min_length):
        self.line_list = []
        for lineOne in self.line_list:
            if len(lineOne.all_point_list) >= min_length:
                self.line_list.append(lineOne)

    def visualize_lines(self, height, width):
        copied_img = np.zeros((height, width, 1), dtype=np.uint8)

        copied_img = cv2.cvtColor(copied_img, cv2.COLOR_GRAY2BGR)
        temp = np.zeros((height, width, 3), dtype=np.uint8)

        for lineOne in self.line_list:
            temp = temp * 0
            for point in lineOne.all_point_list:
                x, y = point
                temp[y][x] = lineOne.color

            copied_img = copied_img + temp

        return copied_img + temp

    def sort(self):
        line_list = copy.deepcopy(self.line_list)

        def criteria(lineOne):
            return len(lineOne.all_point_list)

        line_list.sort(reverse=True, key=criteria)
        self.line_list = line_list

    def leave_long_lines(self, number_of_lines_to_leave=10):
        if number_of_lines_to_leave > len(self.line_list):
            print(
                "Variable \"number_of_lines_to_leave\" is larger than the number of existing lines so execution of this function \"leave_long_lines\" has canceled.")
            return None
        self.sort()
        self.line_list = self.line_list[:number_of_lines_to_leave]

    def is_connectable(self, lineOne1, lineOne2, max_distance):
        for point1 in lineOne1.all_point_list:
            for point2 in lineOne2.all_point_list:
                distance = utils.distance_between(point1, point2)
                if distance < max_distance:
                    return True
        return False

    # TODO change this stupid function's and variables' name
    def pre_combine(self, max_distance):
        new_line_list = []
        line_list = copy.deepcopy(self.line_list)

        for lineOne1 in line_list:
            new_unique_num = lineOne1.unique_num
            recursive_same_indices = []
            for lineOne2 in new_line_list:
                if self.is_connectable(lineOne1, lineOne2, max_distance):
                    same_indices = utils.find_indices(new_line_list,
                                                      lambda index, value:
                                                      value.unique_num == lineOne2.unique_num)
                    recursive_same_indices += same_indices

            for index in recursive_same_indices:
                new_line_list[index].set_unique_num_to(new_unique_num)

            new_line_list.append(lineOne1)
        return new_line_list
        # for i in range(len(line_list)):
        #     for j in range(len(new_line_list)):
        #         if self.is_connectable(self.line_list[i]['hline'], new_line_list[j]['hline'], min_distance):
        #             index_list_of_same_index = utils.find_indices(new_line_list,
        #                                                           lambda index, value: value['index'] ==
        #                                                                                new_line_list[j][
        #                                                                                    'index'])
        #
        #             for index_of_same_index in index_list_of_same_index:
        #                 new_line_list[index_of_same_index]['index'] = self.line_list[i]['index']
        #             # new_line_list[j]['index'] = self.line_list[i]['index']

        # a = 6
        # a1 = 0
        # a1 = round(a * 0.25)
        # a1 = round(a * 0.5)
        # a1 = round(a * 0.75)
        # a1 = a
        #
        # i = 0
        # while True:
        #     lineOne = line_list[i]
        #     line_len = len(lineOne.all_point_list)
        #
        #     first_idx = 0
        #     one_fourth_idx = round(line_len * (1 / 4)) - 1
        #     middle_idx = round(line_len * (1 / 2)) - 1
        #     three_fourth_idx = round(line_len * (3 / 4)) - 1
        #     last_idx = line_len - 1
        #     indices = list({first_idx, one_fourth_idx, middle_idx, three_fourth_idx, last_idx})

    def combine_absolutely(self, new_line_list):
        line_list = copy.deepcopy(new_line_list)
        unique_nums = []
        combined_absolutely = []
        for lineOne in line_list:
            if unique_nums.count(lineOne.unique_num) == 0:
                unique_nums.append(lineOne.unique_num)
                combined_absolutely.append(lineOne)
            else:
                index = unique_nums.index(lineOne.unique_num)
                combined_absolutely[index].all_point_list.extend(lineOne.all_point_list)

        # for i in range(len(new_line_list)):
        #     if unique_nums.count(new_line_list[i].unique_num) == 0:
        #         unique_nums.append(new_line_list[i].unique_num)
        #         combined_absolutely.append([])
        # for i in range(len(new_line_list)):
        #     index = unique_nums.index(new_line_list[i]['index'])
        #     combined_absolutely[index].extend(new_line_list[i]['hline'])
        return combined_absolutely

    # TODO change variable name max_distance to something another
    def combine(self, max_distance):
        pre_combined = self.pre_combine(max_distance)
        absolutely_combined = self.combine_absolutely(pre_combined)
        self.line_list = absolutely_combined

    def imshow(self, title, height, width):
        img = self.visualize_lines(height, width)
        cv2.imshow(title, img)
