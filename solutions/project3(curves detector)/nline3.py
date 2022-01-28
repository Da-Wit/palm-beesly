# Legacy
import cv2
from lineone import *
import copy
import numpy as np
import solutions.utils as utils

image_path = "/test_img/edit6.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img2 = copy.deepcopy(img)
img3 = copy.deepcopy(img)
img4 = copy.deepcopy(img)

h, w = img.shape[:2]
nline = []  # 가로 선 나올때마다 추가됨
nline2 = []  # 세로 선 나올때마다 추가됨

# 맨 처음 for 문 도는데 처음이 선임
# 맨 처음 for 문 도는데 처음이 선이 아님
# 선 인식 됐다가 인식 안된 상황
# 선 인식 안 됐다가 인식 된 상황
# 1번째 for문 바뀌는 상황

img2 = img2 * 0
img3 = img3 * 0
img4 = img4 * 0
# Default value is 0
MIN_GRAYSCALE_VALUE_TO_BE_IDENTIFIED_AS_LINE = 10

# The minimum number of dots in one line
# Default value is 4
MIN_LINE_LENGTH = 10
#  가로 선 찾기
for i in range(0, w - 1):
    find = False
    for j in range(0, h - 1):
        if img[j][i] > MIN_GRAYSCALE_VALUE_TO_BE_IDENTIFIED_AS_LINE and find is False:
            find = True
            set_line_info(nline, i, j)
            # if set_line_info(nline, i, j) is True:
            #     print(",", i, j, end='')  # 기존선에 점추가
            # else:
            #     print("\n새로운선[", i, j, "]")  # 새로운 선 발견
        if img[j][i] == MIN_GRAYSCALE_VALUE_TO_BE_IDENTIFIED_AS_LINE and find is True:
            find = False

#  세로 선 찾기
for j in range(0, h - 1):
    find = False
    for i in range(0, w - 1):
        if img[j][i] > MIN_GRAYSCALE_VALUE_TO_BE_IDENTIFIED_AS_LINE and find is False:
            find = True
            set_line_info(nline2, i, j)
        if img[j][i] == MIN_GRAYSCALE_VALUE_TO_BE_IDENTIFIED_AS_LINE and find is True:
            find = False


def is_connected_hlines(hline1, hline2, min_distance):
    for point1 in hline1:
        for point2 in hline2:
            distance = utils.distance_between(point1, point2)
            if distance < min_distance:
                return True
    return False


def combine_hlines(hline1, hline2):
    combined = np.concatenate((hline1, hline2))
    combined = np.unique(combined, axis=0)
    return combined


def add_hline_to_unnamed_list(unnamed_list, hline1_param, hline2_param, index_param):
    already_exist = len(list(filter(lambda i: i['index'] == index_param, unnamed_list))) == 1
    if already_exist:
        index_of_hline1_in_unnmaed = utils.find_index(unnamed_list,
                                                      lambda index, value: value['index'] == index_param)
        unnamed_list[index_of_hline1_in_unnmaed]['hline'] = combine_hlines(
            unnamed_list[index_of_hline1_in_unnmaed]['hline'], hline2_param)
    else:
        unnamed_list.append({
            'hline': combine_hlines(hline1_param, hline2_param),
            'index': index_param
        })


def combine_unnamed_list_self(unnamed_list):
    new_unnamed = []
    min_distance = 5
    for i in range(len(unnamed_list)):
        for j in range(len(new_unnamed)):
            if is_connected_hlines(unnamed_list[i]['hline'], new_unnamed[j]['hline'], min_distance):
                index_list_of_same_index = utils.find_indices(new_unnamed,
                                                              lambda index, value: value['index'] == new_unnamed[j][
                                                                  'index'])

                for index_of_same_index in index_list_of_same_index:
                    new_unnamed[index_of_same_index]['index'] = unnamed_list[i]['index']
                # new_unnamed[j]['index'] = unnamed_list[i]['index']

        new_unnamed.append(unnamed_list[i])
    return new_unnamed


def merge_same_index_nlines(unnamed_list):
    index_list_of_index = []
    merged = []
    for i in range(len(unnamed_list)):
        if index_list_of_index.count(unnamed_list[i]['index']) == 0:
            index_list_of_index.append(unnamed_list[i]['index'])
            merged.append([])
    for i in range(len(unnamed_list)):
        index = index_list_of_index.index(unnamed_list[i]['index'])
        merged[index].extend(unnamed_list[i]['hline'])
    return merged


def get_img4(nline3, img_for_measurement):
    dst = copy.deepcopy(img_for_measurement) * 0
    for index in range(len(nline3)):
        for x, y in nline3[index]:
            dst[y][x] = 255
        # cv2.imshow("img4", dst)
        # print(len(nline3[index]))
        #
        # k = cv2.waitKey(0)
        # if k == 27:  # Esc key to stop
        #     cv2.destroyAllWindows()
        #     exit(0)
    return dst


nline_copied = copy.deepcopy(nline)
unnamed = []

min_distance = 5

for index_of_horizontal_hline_in_horizontal_nline in range(len(nline_copied)):
    for hline_in_vertical_nline in nline2:
        if is_connected_hlines(nline_copied[index_of_horizontal_hline_in_horizontal_nline].all_point_list,
                               hline_in_vertical_nline.all_point_list,
                               min_distance):
            add_hline_to_unnamed_list(unnamed,
                                      nline_copied[index_of_horizontal_hline_in_horizontal_nline].all_point_list,
                                      hline_in_vertical_nline.all_point_list,
                                      index_of_horizontal_hline_in_horizontal_nline)

for i in nline[:]:
    if len(i.all_point_list) < MIN_LINE_LENGTH:
        nline.remove(i)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.

for i in nline2[:]:
    if len(i.all_point_list) < MIN_LINE_LENGTH:
        nline2.remove(i)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.

unnamed = combine_unnamed_list_self(unnamed)
unnamed = merge_same_index_nlines(unnamed)

nline3 = []

for i in range(len(unnamed)):
    if len(unnamed[i]) >= MIN_LINE_LENGTH:
        nline3.append(unnamed[i])

img4 = get_img4(nline3, img4)

# === 검출된 선 시각적으로 표시================================

# print("세로방향 ", len(nline2), " 개")

for i in nline:
    for k in i.all_point_list:
        img2[k[1]][k[0]] = 255
    # cv2.imshow("img2", img2)
    #
    # k = cv2.waitKey(0)
    # if k == 27:  # Esc key to stop
    #     cv2.destroyAllWindows()
    #     exit(0)

for i in nline:
    # img2[i.pointlist[0][1]][i.pointlist[0][0]] = [0, 255, 255]
    # img2[i.pointlist[-1][1]][i.pointlist[-1][0]] = [255, 0, 255]
    img2[i.all_point_list[0][1]][i.all_point_list[0][0]] = 255
    img2[i.all_point_list[-1][1]][i.all_point_list[-1][0]] = 255

for i in nline2:
    for k in i.all_point_list:
        img3[k[1]][k[0]] = 255
    # cv2.imshow("img3", img3)
    #
    # k = cv2.waitKey(0)
    # if k == 27:  # Esc key to stop
    #     cv2.destroyAllWindows()
    #     exit(0)

for i in nline2:
    # img3[i.pointlist[0][1]][i.pointlist[0][0]] = [0, 255, 255]
    # img3[i.pointlist[-1][1]][i.pointlist[-1][0]] = [255, 0, 255]
    img3[i.all_point_list[0][1]][i.all_point_list[0][0]] = 255
    img3[i.all_point_list[-1][1]][i.all_point_list[-1][0]] = 255

result = img2 + img3 + img4

cv2.imshow("vertical", utils.resize(img3, width=400))
cv2.imshow("horizontal", utils.resize(img2, width=400))
cv2.imshow("combined", utils.resize(img4, width=400))
cv2.imshow("RESULT", utils.resize(result, width=400))

cv2.waitKey(0)
