import cv2
import numpy as np
import utils


#
# scharr1은 잔금 많아서 다른 필터 조정하는 것을 우선으로 하자
#
# TODO canny 직접 구현해서 옵션 조정해서 미세한 주름들이 감지하기
# TODO laplacian 옵션 조정가능하게 구현하기
# TODO 어느 필터든 손금 잘 나오게 조절해서 해봐라
# TODO 손금 추출 관련 논문 구글링 OR 구매하기

# REALIZE canny에 로우패스 필터 적용해서 잡음 제거 하려 했음 -> canny 구현 첫 단계가 가우시안 적용이라 로우패스 필터 적용 안해도 됨
# QUESTION canny가 가장 기본적인 hpf 필터인가? 더 단순한 필터는 없나? -> 가장 단순하진 않음. 경사도만으로 구현하는 SOBEL, SCHARR와 다르게 4단계를 거쳐 구현되기 때문에

# DONE sobel, scharr1, scharr2에 가우시안 적용해서 잡음 제거
# DONE sobel, scharr1, scharr2 옵션 조정하기
# DONE 변수값으로 미세 조정 가능한 하이 패스 필터 구현해보기


def canny(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    equalized = cv2.equalizeHist(denoised)
    blur = cv2.GaussianBlur(equalized, (9, 9), 0)
    min_thresholding = 100
    max_thresholding = 200
    return cv2.Canny(blur, threshold1=min_thresholding, threshold2=max_thresholding)


def using_canny(img):
    cny = canny(img)
    cv2.imshow("canny", cny)


def using_laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_8U)
    cv2.imshow("laplacian", laplacian)
