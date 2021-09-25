# import cv2
# import something
# import utils
# import numpy as np
# import mediapipe as mp
# import math
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# #
# # for num in range(9):
# #
# #     if num == 0:
# #         continue
# #
# #     image_path = f"C:/Users/USER/workspace/palm/images/sample{num}.png"
# #     image = cv2.imread(image_path)
# #     image = utils.remove_bground(image)
# #     image = utils.resize(image, height=600)
# #
# #     landmark = utils.get_hand_form(image, mp_hands)
# #     thumb_outside, thumb_mid, thumb_inside, wrist, middle_between_thumb_wrist, pinky, ring, middle, index, index_inside = something.get_coords(
# #     landmark, image)
# #     cnt = utils.get_contour(image)
# #
# #     aws4 = utils.aws(image, cnt, thumb_outside, index_inside)
# #     aws5 = utils.aws(image, cnt, thumb_mid, index_inside)
# #     aws6 = utils.aws(image, cnt, thumb_inside, index_inside)
# #
# #     cv2.polylines(image, [cnt], True, (255,255,0),2)
# #
# #     # cv2.circle(image,thumb_outside,10,(0,0,255),3)
# #     # cv2.circle(image,thumb_mid,10,(0,0,255),3)
# #     cv2.circle(image,thumb_inside,10,(0,0,255),3)
# #
# #     # cv2.circle(image,aws4,10,(0,0,255),3)
# #     # cv2.circle(image,aws5,10,(0,0,255),3)
# #     cv2.circle(image,aws6,10,(0,0,255),3)
# #
# #     cv2.imshow("img",image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
#
#
#
#
# # a = np.array([6,0])
# # b = np.array([0,0])
# # c = np.array([0,6])
#
# # a = np.array([121,510])
# # b = np.array([136,357])
# # c = np.array([136,457])
# # # result was 5.599339
# #
# # ba = a - b
# # bc = c - b
# #
# # print("ba",ba)
# # print("bc",bc)
# #
# # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
# # angle = np.arccos(cosine_angle)
# # print(np.degrees(angle))
#
#
# def angle3pt(a, b, c):
#     """Counterclockwise angle in degrees by turning from a to c around b
#         Returns a float between 0.0 and 360.0"""
#     ang = math.degrees(
#         math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
#     return ang + 360 if ang < 0 else ang
#
#
# def rotate_point(point, center, degree):
#     deg = (math.pi/180)*degree
#     x1, y1 = point
#     x0, y0 = center
#     x2 = ((x1 - x0) * math.cos(deg)) - ((y1 - y0) * math.sin(deg)) + x0
#     y2 = ((x1 - x0) * math.sin(deg)) + ((y1 - y0) * math.cos(deg)) + y0
#     print(x2, y2)
#
#
# # print(angle3pt((5, 0), (0, 0), (0, 5)))
# # print(angle3pt((136,457), (136,357), (121,510)))
# rotate_point((136,0), (0,0),90)
# # rotate_point((9,0), (0,0),(math.pi/180)*360)
# # print(angle3pt((0,0), (0,0), (121-136,510-357)))


import cv2
import using_mp
import utils
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

number_of_sample_images = 44


def palm_center(palm_landmarks):
    lands = palm_landmarks.copy()
    palm_except_fingers = np.array([lands[0],   # WRIST
                                    lands[5],   # INDEX_FINGER_MCP
                                    lands[9],   # MIDDLE_FINGER_MCP
                                    lands[13],  # RING_FINGER_MCP
                                    lands[17],  # PINKY_MCP
                                    ])
    return utils.center(palm_except_fingers)


def get_point_of_rotated_vertices(width, height, degree, center):
    left_top = (0, 0)
    right_top = (width - 1, 0)
    left_bottom = (0, height - 1)
    right_bottom = (width - 1, height - 1)

    rotated_left_top = utils.rotate_point(left_top, center, degree)
    rotated_right_top = utils.rotate_point(right_top, center, degree)
    rotated_left_bottom = utils.rotate_point(left_bottom, center, degree)
    rotated_right_bottom = utils.rotate_point(right_bottom, center, degree)

    return rotated_left_top, rotated_right_top, rotated_left_bottom, rotated_right_bottom


def get_rotated_size(width, height, degree, center):
    rotated_left_top, rotated_right_top, rotated_left_bottom, rotated_right_bottom = get_point_of_rotated_vertices(
        width, height, degree, center)

    min_x, max_x, min_y, max_y = utils.minmax_xy(np.array([rotated_left_top,
                                                           rotated_right_top,
                                                           rotated_left_bottom,
                                                           rotated_right_bottom,
                                                           ]))
    rotated_width, rotated_height = round(max_x - min_x), round(max_y - min_y)
    return rotated_width, rotated_height


# 이미지 샘플 개수만큼 for문을 돈다
for i in range(number_of_sample_images):
    image_path = f"C:/Users/USER/workspace/palm/images/sample{i}.png"
    image = cv2.imread(image_path)

    # 이미지가 제대로 불러와지지 않으면 에러 출력하고 다음 숫자로 넘어감
    if image is None:
        print(f"images/sample{i}.png is empty!!")
        continue

    # 출력 시 화면에 적당한 크기로 출력되게 하기 위해 이미지를 resize함
    image = utils.resize(image, height=400)
    img = image.copy()

    width = img.shape[1]
    height = img.shape[0]

    landmarks = np.array(using_mp.get_hand_landmark(img))
    wrist = [landmarks[0][0], landmarks[0][1]]

    center = palm_center(landmarks)

    cv2.circle(img, center, 5, (0, 0, 255), 2)

    cv2.imshow(f"image{i} original", img)

    degree_to_rotate = utils.getAngle(center, wrist) - 90
    rotated_width, rotated_height = get_rotated_size(
        width, height, -degree_to_rotate, center)

    rotated_left_top, rotated_right_top, rotated_left_bottom, rotated_right_bottom = get_point_of_rotated_vertices(
        width, height, -degree_to_rotate, center)

    min_x, max_x, min_y, max_y = utils.minmax_xy(np.array([rotated_left_top,
                                                           rotated_right_top,
                                                           rotated_left_bottom,
                                                           rotated_right_bottom,
                                                           ]))

    M = cv2.getRotationMatrix2D(center, degree_to_rotate, 1.0)

    top_border_size = min_y * -1 if min_y < 0 else 0
    bottom_border_size = max_y - width if max_y - width > 0 else 0
    left_border_size = min_x * -1 if min_x < 0 else 0
    right_border_size = max_x - width if max_x - width > 0 else 0

    _, max_border_size, _, _ = cv2.minMaxLoc(np.array([top_border_size,
                                                      bottom_border_size,
                                                      left_border_size,
                                                      right_border_size,
                                                       ]))
    max_border_size = round(max_border_size)

    img = cv2.copyMakeBorder(
        img,
        top=max_border_size,
        bottom=max_border_size,
        left=max_border_size,
        right=max_border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 255]
    )

    cv2.imshow("extended img", img)

    bordered_width = width + (max_border_size * 2)
    bordered_height = height + (max_border_size * 2)

    rotated_bordered_width, rotated_bordered_height = get_rotated_size(
        bordered_width, bordered_height, -degree_to_rotate, center)

    # rotated_img = cv2.CreateMat(rotated_height, rotated_width, )
    img = cv2.warpAffine(
        img, M, (rotated_bordered_width, rotated_bordered_height))

    cv2.imshow(f"image{i}", img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for문 도중 Esc를 누르면 프로그램이 종료되게 함
    if k == 27:    # Esc key to stop
        break
    elif k == -1:  # normally -1 returned,so don't print it
        continue

cv2.destroyAllWindows()
