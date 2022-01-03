from lineone import LineOne
import random
import copy
import cv2
import solutions.utils as utils


class Lines:
    def __init__(self):
        self.line_list = []
        self.number_of_front_points_to_find_slope = 4

    def add_line(self, line):
        self.line_list.append(line)

    def get_indices_of_nearby_lines(self, point, max_distance):
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
        close_lines = self.get_indices_of_nearby_lines(point, max_distance)
        length_of_close_lines = len(close_lines)

        # 점 주변에 선이 0개일 때
        # 새로운 hline 만들어서 nline에 append
        # 즉, 새로운 선 발견했다고 추정해 새로운 선 추가
        if length_of_close_lines == 0:
            lineOne = LineOne(self.number_of_front_points_to_find_slope)
            self.add_line(lineOne)
            lineOne.add_point(point)

        # 점 주변에 선이 1개일 때
        # 그 1개의 선에 점 추가
        elif length_of_close_lines == 1:
            lineOne = self.line_list[close_lines[0]]
            lineOne.add_point(point)


        # 점 주변에 선이 1개보다 많을 때
        # 기울기로 구함
        else:
            filtered_lines = []

            for i in range(length_of_close_lines):
                hline = self.line_list[close_lines[i]]

                if len(hline.point_list) < self.number_of_front_points_to_find_slope:
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
                self.line_list[close_lines[random_index]].add_point(point)


            else:
                min_gap = filtered_lines[0]['gap']
                min_gap_idx = 0
                for i in range(1, len(filtered_lines)):
                    if min_gap > filtered_lines[i]['gap']:
                        min_gap = filtered_lines[i]['gap']
                        min_gap_idx = i

                self.line_list[min_gap_idx].add_point(point)

    def filter_by_line_length(self, min_length):
        line_list = copy.deepcopy(self.line_list)
        for lineOne in line_list:
            if len(lineOne.point_list) < min_length:
                line_list.remove(lineOne)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.
        self.line_list = line_list

    def visualize_lines(self, img_param, imshow=False, color=False):
        copied_img = copy.deepcopy(img_param)
        if color:
            copied_img = cv2.cvtColor(copied_img, cv2.COLOR_GRAY2BGR)

        for lineOne in self.line_list:
            for point in lineOne.point_list:
                x = point[0]
                y = point[1]

                if color:
                    copied_img[y][x] = lineOne.color
                else:
                    copied_img[y][x] = 255

            if imshow:
                cv2.imshow("img_on_progress", utils.resize(copied_img, width=600))

                k = cv2.waitKey(0)
                if k == 27:  # Esc key to stop
                    cv2.destroyAllWindows()
                    exit(0)
        return copied_img
