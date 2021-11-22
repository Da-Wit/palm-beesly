import cv2
import numpy as np

# scharr 필터 마스크는 'opencv 4로 배우는 컴퓨터 비전과 머신 러닝'이라는 책을 참조함
# scharr 5x5 필터 마스크는 다음 글에서 참조함: https://www.yumpu.com/en/document/read/51859112/sobel-and-scharr-5x5-gradients-classical-image-processing-library

# TODO 상수를 최상위의 한 구역에서 관리하도록 바꿔야함
SC3X3 = 'scharr_3x3'
SC5X5 = 'scharr_5x5'
SC7X7 = 'scharr_7x7'
X = 'x'
Y = 'y'

SCHARR_KSIZE = {
    SC3X3: {X: np.array([[-3, 0, 3],
                        [-10, 0, 10],
                        [-3, 0, 3]]) / 60,
            Y: np.array([[-3, -10, -3],
                        [0, 0, 0],
                        [3, 10, 3]]) / 60},
    SC5X5: {X: np.array([
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
        ]) / 60},
    SC7X7: {X: np.array([
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
        ])},
}


def process_scharr_trackbar_val(val):
    # 트랙바는 0부터 n까지의 정수만 선택할 수 있기 때문에
    # 소숫점에 접근하기 위해 다음과 같이 val을 나눠준다.
    alpha = val / 10
    return alpha


def process_gaussian_trackbar_val(val):
    ksize = val * 2 + 1
    return ksize


def calculate_scharr(img, alpha, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 소벨 필터값
    x_kernel = SCHARR_KSIZE[ksize][X]
    y_kernel = SCHARR_KSIZE[ksize][Y]

    # '트랙바의 값 / 10'을 소벨 필터값 전체에 곱한다
    x_kernel = alpha * x_kernel
    y_kernel = alpha * y_kernel

    # 이미지에 소벨 필터를 적용한다
    edge_gx = cv2.filter2D(gray, -1, x_kernel)
    edge_gy = cv2.filter2D(gray, -1, y_kernel)

    merged = edge_gx + edge_gy

    return merged


def calculate_gaussian(img, ksize):
    gaussian = cv2.GaussianBlur(img, (ksize, ksize), cv2.BORDER_DEFAULT)


def using_scharr(img, scharr_ksize=SC3X3):
    if not scharr_ksize in SCHARR_KSIZE.keys():
        print('Given kernel size is not correct.')
        exit(0)

    window_name = scharr_ksize
    scharr_trackbar_name = 'scharr'
    gaussian_trackbar_name = 'gaussian'
    max_scharr_trackbar_val = 500
    max_gaussian_trackbar_val = 10
    scharr_trackbar_start_pos = 0
    gaussian_trackbar_start_pos = 0

    def on_scharr_trackbar_changed(trackbar_val, image):
        # 트랙바 값을 처리해서 실제 사용할 alpha를 얻는다
        alpha = process_scharr_trackbar_val(trackbar_val)

        # 소벨 필터가 적용된 이미지를 구함
        scharr_ed = calculate_scharr(image, alpha, scharr_ksize)

        # 가우시안 트랙바 값 가져옴
        gaussian_trackbar_val = cv2.getTrackbarPos(
            gaussian_trackbar_name, window_name)

        # 가우시안 트랙바 값으로 커널 사이즈 구함
        gaussian_ksize = process_gaussian_trackbar_val(gaussian_trackbar_val)

        # 소벨ed 이미지에 가우시안 적용함
        result_img = cv2.GaussianBlur(
            scharr_ed, (gaussian_ksize, gaussian_ksize), cv2.BORDER_DEFAULT)

        # 소벨ed에 가우시안 적용된 이미지 출력
        cv2.imshow(window_name, result_img)

    def on_gaussian_trackbar_changed(trackbar_val, image):
        scharr_trackbar_val = cv2.getTrackbarPos(
            scharr_trackbar_name, window_name)

        alpha = process_scharr_trackbar_val(scharr_trackbar_val)

        scharr_ed = calculate_scharr(image, alpha, scharr_ksize)

        gaussian_ksize = process_gaussian_trackbar_val(trackbar_val)

        result_img = cv2.GaussianBlur(
            scharr_ed, (gaussian_ksize, gaussian_ksize), cv2.BORDER_DEFAULT)

        cv2.imshow(window_name, result_img)

    # window_name을 이름으로 하는 윈도우를 만들어 놓음으로써 해당 윈도우에 트랙바를 달 수 있게 함
    cv2.namedWindow(window_name)

    # 이름이 window_name인 창에 scharr 커널에 곱할 값을 설정하는 트랙바를 만든다
    cv2.createTrackbar(scharr_trackbar_name,
                       window_name,
                       scharr_trackbar_start_pos,
                       max_scharr_trackbar_val,
                       lambda val: on_scharr_trackbar_changed(val, img),
                       )

    # 이름이 window_name인 창에 가우시안 커널 크기를 설정하는 트랙바를 만든다
    cv2.createTrackbar(gaussian_trackbar_name,
                       window_name,
                       gaussian_trackbar_start_pos,
                       max_gaussian_trackbar_val,
                       lambda val: on_gaussian_trackbar_changed(val, img),
                       )

    on_scharr_trackbar_changed(scharr_trackbar_start_pos, img)
