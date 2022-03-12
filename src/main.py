import cv2 as cv
from lines import Lines
from timeit import default_timer as timer
from copy import deepcopy as cp
import constants
import utils
import numpy as np
import platform
from roi import get_palm_roi
from hpf import HPF
from hpf import HPF_TYPE
from plantcv import plantcv as pcv

# My Specifications
print(f"platform.platform() : {platform.platform()}")  # macOS-12.2.1-x86_64-i386-64bit
print(f"platform.release() : {platform.release()}")  # 21.3.0
print(
    f"platform.version() : {platform.version()}")  # Darwin Kernel Version 21.3.0: Wed Jan  5 21:37:58 PST 2022; root:xnu-8019.80.24~20/RELEASE_ARM64_T8101
print(f"python version : {platform.python_version()}")  # 3.8.12
print(f"opencv version : {cv.__version__}")  # 4.5.5
print(f"numpy version : {np.__version__}")  # 1.22.2


# 한 방향으로의 hline들의 리스트, nline을 리턴함
def find_one_orientation_lines(img_param, max_line_distance, is_horizontal):
    height, width = img_param.shape[:2]

    if is_horizontal:
        first_for_loop_max = width
        second_for_loop_max = height
    else:
        first_for_loop_max = height
        second_for_loop_max = width

    lines = Lines()
    unique_num = 0
    for i in range(first_for_loop_max):
        if is_horizontal:
            x = i
        else:
            y = i
        for j in range(second_for_loop_max):
            if is_horizontal:
                y = j
            else:
                x = j

            if img_param[y][x] == 255:
                lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal)
                unique_num += 1
        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        # 2중 for문이 돌 때마다 실행한다.
        # lines에 저장된 모든 선들의 모든 점들에 대해 연산을 수행할 필요는 없다.
        # 그래서 작업 영역(max_line_distance)를 벗어난 점들을 작업 중에만 제거해준다.
        lines.renew_work_area([x, y], max_line_distance, is_horizontal)

    return lines


# Now it's same with find_one_orientation_lines
# Please update this function too, when updating function find_one_orientation_lines
def find_one_orientation_lines_with_debugging(img_param, max_line_distance, is_horizontal):
    height, width = img_param.shape[:2]
    img_for_debugging = np.zeros((height, width, 1), dtype=np.uint8)
    img_for_debugging = cv.cvtColor(img_for_debugging, cv.COLOR_GRAY2BGR)

    if is_horizontal:
        first_for_loop_max = width
        second_for_loop_max = height
    else:
        first_for_loop_max = height
        second_for_loop_max = width

    lines = Lines()
    unique_num = 0
    skipped = False
    num_of_skip = 40

    for i in range(first_for_loop_max):
        if is_horizontal:
            x = i
        else:
            y = i

        for j in range(second_for_loop_max):
            if skipped:
                num_of_skip -= 1

            if is_horizontal:
                y = j
            else:
                x = j

            if skipped:
                if img_param[y][x] == 255:
                    lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal, debug=True,
                                       img_for_debugging=img_for_debugging)
                    unique_num += 1
                if num_of_skip == 0:
                    skipped = False
                    img_for_debugging *= 0
                    for lineOne in lines.line_list:
                        for point in lineOne.all_point_list:
                            x2, y2 = point
                            img_for_debugging[y2][x2] = lineOne.color
            else:
                if img_param[y][x] == 255:
                    lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal, debug=True,
                                       img_for_debugging=img_for_debugging)
                    unique_num += 1
                prev_bgr = cp(img_for_debugging[y][x])
                img_for_debugging[y][x] = [0, 0, 255]
                cv.imshow("debugging", img_for_debugging)
                k = cv.waitKey(0)
                # for문 도중 Esc를 누르면 프로그램이 종료되게 함
                if k == 113:  # q key to stop
                    exit(0)
                if k == 115:  # s key to skip
                    skipped = True
                    num_of_skip = 20
                    img_for_debugging[y][x] = prev_bgr
                else:
                    img_for_debugging[y][x] = prev_bgr

        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        lines.renew_work_area([x, y], max_line_distance, is_horizontal)

    return lines


def init_imgs():
    image_path = "/Users/david/workspace/palm-beesly/sample_img/sample7.png"
    image = cv.imread(image_path)

    if image is None:
        print(constants.ERROR_MESSAGE["IMG_IS_EMPTY"])
        exit(1)
    return image


def thin(bgr_img):
    gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    result = utils.thin(thresh)  # to use this function paste it: pip install opencv-contrib-python
    return result


if __name__ == "__main__":
    original = init_imgs()
    roi, rect = get_palm_roi(original)
    x, y, w, h = rect
    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    hpfed = HPF(HPF_TYPE.SC3X3, alpha=500, gaussian=4).process(roi)
    thr = utils.adaptive_threshold(hpfed, 11, 0)

    thinned = utils.thin(thr)
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=thinned, size=35)
    cv.imshow("pruned_skeleton", pruned_skeleton)

    max_line_distance = 2  # Default value is 3
    combining_distance = 1
    min_line_length = 5  # The minimum number of dots in one line, Default value is 4
    number_of_lines_to_leave = 6  # Default value is 10

    height, width = pruned_skeleton.shape[:2]

    start = timer()
    # lines = find_one_orientation_lines_with_debugging(thr, max_line_distance, is_horizontal=False)
    lines = find_one_orientation_lines(pruned_skeleton, max_line_distance, is_horizontal=False)
    lines.imshow("just found", height, width)
    stop = timer()
    print("Finding part", round(stop - start, 6))

    start = timer()
    lines.filter_by_line_length(min_line_length)
    # lines.sort()
    # del lines.line_list[0]
    # lines.leave_long_lines(number_of_lines_to_leave)
    filtered = lines.visualize_lines(height, width)
    cv.imshow("filtered", filtered)
    stop = timer()
    print("Filtering part", round(stop - start, 6))
    print("Done")

    cv.waitKey(0)
