import cv2
import numpy as np
import enum

# 구현 할 때 사용한 스펙
# cv2 버젼(cv2.__version__) -> 4.5.2
# numpy 버젼(np.__version__) -> 1.19.5
# 파이썬 버젼(python --version) -> 3.9.5


# 그냥 상수들
X = 'x'
Y = 'y'
MAX_ALPHA = 'max_alpha'
DIVIDER = 'divider'

# process 함수를 실행할 때 인풋으로 alpha나 gaussian이 주어지지 않았을 때 해당 인풋의 기본값으로 사용될 변수
# 값이 주어졌는지 여부를 if로 확인하여 값이 주어졌을 때에만 후처리를 하도록 하기 위해 사용함
NOT_FOUND = -404

# HPF를 sobel로써 사용하려 한다면 alpha에 0~1000 값만 사용할 수 있음
# alpha는 트랙바에서 sobel이나 scharr에 해당함
# 사용 예시는 밑에 참고
max_sobel_alpha = 1000
sobel_divider = 750

# HPF를 sobel로써 사용하려 한다면 alpha에 0~500 값만 사용할 수 있음
max_scharr_alpha = 500
scharr_divider = 10

# HPF 클래스로 사용할 수 있는 소벨과 샤르 필터 모두 하나의 가우시안 최대값을 공통으로 사용하여 아래처럼 정의함
MAX_GAUSSIAN_VALUE = 10


class HPF_type(enum.IntEnum):
    SB3X3 = 0
    SB5X5 = 1
    SB7X7 = 2
    SC3X3 = 3
    SC5X5 = 4


# 각 필터들에 해당하는 데이터
HPF_kernel = {
    HPF_type.SB3X3: {X: np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),
                     Y: np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]),
                     MAX_ALPHA: max_sobel_alpha,
                     DIVIDER: sobel_divider
                     },
    HPF_type.SB5X5: {X: np.array([
        [-5, -4, 0, 4, 5],
        [-8, -10, 0, 10, 8],
        [-10, -20, 0, 20, 10],
        [-8, -10, 0, 10, 8],
        [-5, -4, 0, 4, 5]]) / 20,
        Y: np.array([
            [-5, -8, -10, -8, -5],
            [-4, -10, -20, -10, -4],
            [0, 0, 0, 0, 0],
            [4, 10, 20, 10, 4],
            [5, 8, 10, 8, 5]
        ]) / 20,
        MAX_ALPHA: max_sobel_alpha,
        DIVIDER: sobel_divider
    },
    HPF_type.SB7X7: {X: np.array([
        [-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18],
        [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
        [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
        [-3 / 9, -2 / 4, -1 / 1, 0, 1 / 1, 2 / 4, 3 / 9],
        [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
        [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
        [-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18]]),
        Y:  np.array([
            [-3 / 18, -3 / 13, -3 / 10, -3 / 9, -3 / 10, -3 / 13, -3 / 18],
            [-2 / 13, -2 / 8, -2 / 5, -2 / 4, -2 / 5, -2 / 8, -2 / 13],
            [-1 / 10, -1 / 5, -1 / 2, -1 / 1, -1 / 2, -1 / 5, -1 / 10],
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 10, 1 / 5, 1 / 2, 1 / 1, 1 / 2, 1 / 5, 1 / 10],
            [2 / 13, 2 / 8, 2 / 5, 2 / 4, 2 / 5, 2 / 8, 2 / 13],
            [3 / 18, 3 / 13, 3 / 10, 3 / 9, 3 / 10, 3 / 13, 3 / 18]
        ]),
        MAX_ALPHA: max_sobel_alpha,
        DIVIDER: sobel_divider
    },

    HPF_type.SC3X3: {X: np.array([[-3, 0, 3],
                                  [-10, 0, 10],
                                  [-3, 0, 3]]) / 60,
                     Y: np.array([[-3, -10, -3],
                                  [0, 0, 0],
                                  [3, 10, 3]]) / 60,
                     MAX_ALPHA: max_scharr_alpha,
                     DIVIDER: scharr_divider
                     },
    HPF_type.SC5X5: {X: np.array([
        [-1, -1, 0, 1, 1],
        [-2, -2, 0, 2, 2],
        [-3, -6, 0, 6, 3],
        [-2, -2, 0, 2, 2],
        [-1, -1, 0, 1, 1]]) / 60,
        Y: np.array([
            [-1, -2, -3, -2, -1],
            [-1, -2, -6, -2, -1],
            [0, 0, 0, 0, 0],
            [1, 2, 6, 2, 1],
            [1, 2, 3, 2, 1]
        ]) / 60,
        MAX_ALPHA: max_scharr_alpha,
        DIVIDER: scharr_divider},
}


class HPF:
    def check_alpha(self, alpha, type):
        if alpha < 0:
            return 'Alpha argument should be bigger than 0.'

        if alpha > self.kernel[MAX_ALPHA]:
            return f'Alpha argument of type {type.name} should be lower than max({self.kernel[MAX_ALPHA]}).'

        return None

    def check_gaussian(self, gaussian):
        if gaussian < 0:
            return 'gaussian argument should be bigger than 0.'
        if gaussian > MAX_GAUSSIAN_VALUE:
            return f'gaussian argument should be lower than {MAX_GAUSSIAN_VALUE}.'

        return None

    def __init__(self, type, alpha, gaussian):
        # 예외처리
        if not isinstance(type, HPF_type):
            print('Given HPF type is not supported.')
            exit(0)

        self.kernel = HPF_kernel[type]

        alpha_checker = self.check_alpha(alpha, type)
        gaussian_checker = self.check_gaussian(gaussian)

        if not alpha_checker == None:
            print(alpha_checker)
            exit(0)
        if not gaussian_checker == None:
            print(gaussian_checker)
            exit(0)

        # 트랙바에 해당하는 값들을 실제 필터링, 가우시안 블러에 사용하는 값으로 연산해서 클래스에 저장
        self.alpha = alpha / self.kernel[DIVIDER]
        self.gaussian_ksize = (gaussian * 2) + 1

    def calculate_HPF(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 소벨 필터값
        x_kernel = self.kernel[X]
        y_kernel = self.kernel[Y]

        # '트랙바의 값 / 10'을 소벨 필터값 전체에 곱한다
        x_kernel = self.alpha * x_kernel
        y_kernel = self.alpha * y_kernel

        # 이미지에 소벨 필터를 적용한다
        edge_gx = cv2.filter2D(gray, -1, x_kernel)
        edge_gy = cv2.filter2D(gray, -1, y_kernel)

        default_weight = 1.0
        merged = cv2.addWeighted(
            edge_gx, default_weight, edge_gy, default_weight, 0)

        return merged

    def calculate_gaussian(self, img):
        gaussian = cv2.GaussianBlur(
            img, (self.gaussian_ksize, self.gaussian_ksize), cv2.BORDER_DEFAULT)
        return gaussian

    def process(self, img, alpha=NOT_FOUND, gaussian=NOT_FOUND):
        if alpha != NOT_FOUND:
            alpha_checker = self.check_alpha(alpha, type)

            if not alpha_checker == None:
                print(alpha_checker)
                exit(0)

            self.alpha = alpha / self.kernel[DIVIDER]

        if gaussian != NOT_FOUND:
            gaussian_checker = self.check_gaussian(gaussian)

            if not gaussian_checker == None:
                print(gaussian_checker)
                exit(0)

            self.gaussian_ksize = (gaussian * 2) + 1

        HPF_calculated = self.calculate_HPF(img)
        result_img = self.calculate_gaussian(HPF_calculated)
        return result_img


# 아래는 사용 예시입니다

# 이미지 읽기
image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)

# alpha(HPF의 2번째 인자)는 트랙바에서처럼 필터마다 최대값이 다른고 0 ~ max_sobel_alpha 혹은 0 ~ max_scharr_alpha이어야 함
# gaussian(HPF의 3번째 인자)은 트랙바에서처럼 0~10만 가능함

sb3x3 = HPF(HPF_type.SB3X3, alpha=500, gaussian=1).process(img)
cv2.imshow('sb3x3', sb3x3)

sb5x5 = HPF(HPF_type.SB5X5, alpha=500, gaussian=2)
cv2.imshow('sb5x5', sb5x5.process(img))

sb7x7 = HPF(HPF_type.SB7X7, alpha=500, gaussian=3)
sb7x7_result = sb7x7.process(img)
cv2.imshow('sb7x7', sb7x7_result)

sc3x3 = HPF(HPF_type.SC3X3, alpha=500, gaussian=4).process(img)
cv2.imshow('sc3x3', sc3x3)

sc5x5 = HPF(HPF_type.SC5X5, alpha=500, gaussian=5).process(img)
cv2.imshow('sc5x5', sc5x5)


cv2.waitKey(0)
cv2.destroyAllWindows()
