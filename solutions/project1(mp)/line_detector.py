import cv2
import mediapipe as mp
import numpy as np
import using_mp
import solutions.utils as utils

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

number_of_sample_images = 44

# 이미지 샘플 개수만큼 for문을 돈다
for i in range(number_of_sample_images):
    image_path = f"C:/Users/think/workspace/palm-beesly/test_img/sample{3}.png"
    # image_path = f"C:/Users/think/workspace/palm-beesly/test_img/sample{25}.png"
    image = cv2.imread(image_path)

    # 이미지가 제대로 불러와지지 않으면 에러 출력하고 다음 숫자로 넘어감
    if image is None:
        print(f"test_img/sample{i}.png is empty!!")
        continue

    # 출력 시 화면에 적당한 크기로 출력되게 하기 위해 이미지를 resize함
    image = utils.resize(image, height=600)
    img = image.copy()

    width = img.shape[1]
    height = img.shape[0]
    sonnal, coords = using_mp.get_palm(img)

    # # coords에 있는 값들 중 유효하지 않은 값은 [0,0]이 되도록 설정했습니다.
    # # 만약 그 값을 제외하고 사용하고자 한다면(제 생각에 나름 합리적인 방법인)
    # # 아래의 refine_coords를 사용하세요.
    # # 사용 예시는 다음과 같습니다.

    # # sonnal[0] : sonnal_top
    # # sonnal[len(sonnal)-1] : sonnal_bottom
    # cv2.circle(image, sonnal[0][0], 5, (0, 255, 0), 2)
    # cv2.circle(image, sonnal[len(sonnal)-1][0], 5, (0, 255, 0), 2)
    palm = np.append(sonnal, coords, axis=0)
    # img = cv2.polylines(img, [palm], True, (0, 255, 0), 2)

    landmarks = np.array(using_mp.get_hand_landmark(img))

    boundingRect = cv2.boundingRect(landmarks)
    x1, y1, x2, y2 = boundingRect
    # cv2.circle(img, (x1+ round(x2/2), y1), 5, (255, 255, 0), 10)

    wrist = [landmarks[0][0], landmarks[0][1]]
    # 손바닥 중심

    palm_except_fingers = np.array([landmarks[0],  # WRIST
                                    landmarks[5],  # INDEX_FINGER_MCP
                                    landmarks[9],  # MIDDLE_FINGER_MCP
                                    landmarks[13],  # RING_FINGER_MCP
                                    landmarks[17],  # PINKY_MCP
                                    ])

    nucleus = utils.center(palm_except_fingers)

    pts = np.array([wrist, nucleus], np.int32)

    # cv2.circle(img, wrist, 5, (0, 0, 255), 2)
    cv2.circle(img, nucleus, 1, (255, 255, 255), 5)
    cv2.circle(img, nucleus, 5, (0, 0, 255), 2)

    cv2.rectangle(img, boundingRect, (255, 0, 0), 3)
    # 손의 특정 좌표들을 화면에 표시

    # All landmarks iteration
    for coord in landmarks:
        cv2.circle(img, coord, 1, (255, 255, 255), 4)
        cv2.circle(img, coord, 4, (0, 0, 0), 2)

    # palm_except_fingers iteration
    for coord in palm_except_fingers:
        cv2.circle(img, coord, 1, (255, 255, 255), 4)
        cv2.circle(img, coord, 4, (0, 255, 0), 2)

    cv2.imshow(f"image{i} original", img)
    # cv2.circle(img, (nucleus[0], nucleus[1] + 10), 5, (0, 0, 255), 2)

    degree_to_rotate = utils.getAngle(nucleus, wrist) - 90

    print(degree_to_rotate + 90)

    left_top = (0, 0)
    right_top = (width - 1, 0)
    left_bottom = (0, height - 1)
    right_bottom = (width - 1, height - 1)

    rotated_left_top = utils.rotate_point(left_top, nucleus, -degree_to_rotate)
    rotated_right_top = utils.rotate_point(
        right_top, nucleus, -degree_to_rotate)
    rotated_left_bottom = utils.rotate_point(
        left_bottom, nucleus, -degree_to_rotate)
    rotated_right_bottom = utils.rotate_point(
        right_bottom, nucleus, -degree_to_rotate)

    min_x, max_x, _, _ = cv2.minMaxLoc(np.array([rotated_left_top[0],
                                                 rotated_right_top[0],
                                                 rotated_left_bottom[0],
                                                 rotated_right_bottom[0],
                                                 ]))

    min_y, max_y, _, _ = cv2.minMaxLoc(np.array([rotated_left_top[1],
                                                 rotated_right_top[1],
                                                 rotated_left_bottom[1],
                                                 rotated_right_bottom[1],
                                                 ]))
    rotated_width, rotated_height = round(max_x - min_x), round(max_y - min_y)

    # cv2.putText(img, f"{angle}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
    # pts = pts.reshape((-1, 1, 2));

    M = cv2.getRotationMatrix2D(nucleus, degree_to_rotate, 1.0)

    top_border_size = min_y * -1 if min_y < 0 else 0
    bottom_border_size = max_y - width if max_y - width > 0 else 0
    left_border_size = min_x * -1 if min_x < 0 else 0
    right_border_size = max_x - width if max_x - width > 0 else 0

    horizontal_border_size = round((rotated_width - width) / 2)
    vertical_border_size = round((rotated_height - height) / 2)

    img = cv2.copyMakeBorder(
        img,
        top=round(top_border_size),
        bottom=round(bottom_border_size),
        left=round(left_border_size),
        right=round(right_border_size),
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 255]
    )

    # cv2.imshow("extended img", img)

    # rotated_img = cv2.CreateMat(rotated_height, rotated_width, )
    img = cv2.warpAffine(img, M, (rotated_width, rotated_height))

    # 아래 이 주석은 수직, 수평 테두리 두께를 출력하는 코드임
    hori_text = f"hori: {horizontal_border_size}"
    vert_text = f"vert: {vertical_border_size}"

    text_size = cv2.getTextSize(hori_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    # cv2.putText(img, hori_text, (50,50),cv2.FONT_HERSHEY_SIMPLEX ,
    #                1, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(img, vert_text, (50,50+text_size[0][1]+10),cv2.FONT_HERSHEY_SIMPLEX ,
    #                1, (0,255,0), 2, cv2.LINE_AA)

    # print(text_size[0])
    # cv2.putText(img, vert_text, (50+text_size[0],50+text_size[1]),cv2.FONT_HERSHEY_SIMPLEX ,
    #                1, (0,255,0), 2, cv2.LINE_AA)
    # img = cv2.polylines(img, [pts], False, (0, 255, 0), 2)

    # cv2.imshow(f"image{i}", img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for문 도중 Esc를 누르면 프로그램이 종료되게 함
    if k == 27:  # Esc key to stop
        break
    elif k == -1:  # normally -1 returned,so don't print it
        continue

cv2.destroyAllWindows()
