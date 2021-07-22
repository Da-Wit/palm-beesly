import cv2
import something
import utils
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

image_path = "some path"
image = cv2.imread(image_path)
image = utils.resize(image, height=600)
# TODO 이미지 높이에 관계없이 get_palm 함수가 동작하도록 만들기
# 현재는 이미지 높이가 600일 경우에 최적화되어있고,
# 테스트 또한 높이가 600일 경우에만 진행하여
# 600이 아닐 경우 성능을 보장할 수 없다.

# TODO remove_all_except_hand 함수 구현하기
# remove_bground 함수로 배경을 지우기 전에
# 손 구역을 mediapipe로 알아내서, 그 부분을 제외한
# 나머지 부분을 지워야 한다(검정색으로 만들어야 한다.).
# 그러지 않으면 remove_bground 함수 실행이
# 손을 지울 수 있다.
# 단기적으로는 remove_bground의 값 조절이
# 효율적일지라도, 장기적으로는
# remove_all_except_hand 함수를
# 구현하는 것이 현명할 것이다.


sonnal, coords = something.get_palm(image)

# # coords에 있는 값들 중 유효하지 않은 값은 [0,0]이 되도록 설정했습니다.
# # 만약 그 값을 제외하고 사용하고자 한다면(제 생각에 나름 합리적인 방법인)
# # 아래의 refine_coords를 사용하세요.
# # 사용 예시는 다음과 같습니다.
# coords = something.refine_coords(coords)

# # sonnal, coords 사용 예시
# for coord in coords:
#     cv2.circle(image, coord[0], 5, (0, 255, 0), 2)

# # sonnal[0] : sonnal_top,
# # sonnal[len(sonnal)-1] : sonnal_bottom
# cv2.circle(image, sonnal[0][0], 5, (0, 255, 0), 2)
# cv2.circle(image, sonnal[len(sonnal)-1][0], 5, (0, 255, 0), 2)
# image = cv2.polylines(image, [sonnal], False, (0, 255, 0), 2)


landmarks = something.get_hand_landmark(image)
# # landmarks는 손의 특정 구역들의 좌표를 가진 배열입니다.
# # 예를 들어, PINKY_MCP, WRIST 등이 있습니다.
# # landmarks 사용 예시
# for coord in landmarks:
#     cv2.circle(image, coord, 5, (0, 0, 255), 2)
