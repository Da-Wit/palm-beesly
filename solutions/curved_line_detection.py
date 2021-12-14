import copy
import cv2
from hline import *

image_path = "C:/Users/USER/Downloads/edit4.png"
# image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

cv2.imshow("ORIGINAL", img)
img2 = copy.deepcopy(img)
img3 = copy.deepcopy(img)
h, w = img.shape[:2]
nline = []  # 가로 선 나올때마다 추가됨
nline2 = []  # 세로 선 나올때마다 추가됨

# 맨 처음 for 문 도는데 처음이 선임
# 맨 처음 for 문 도는데 처음이 선이 아님
# 선 인식 됐다가 인식 안된 상황
# 선 인식 안 됐다가 인식 된 상황
# 1번째 for문 바뀌는 상황


#  가로 선 찾기
for i in range(0, w - 1):

    find = False
    # print("-----------------i:", i)

    for j in range(0, h - 1):
        # print("j:", j)
        img2[j][i] = 0
        img3[j][i] = 0
        if img[j][i] != 255 and find is False:
            find = True
            if set_line_info(nline, i, j) is True:
                print(",", i, j, end='')  # 기존선에 점추가
            else:
                print("\n새로운선[", i, j, "]")  # 새로운 선 발견
        if img[j][i] == 255 and find is True:
            find = False

for i in nline[:]:
    if len(i.pointlist) < 4:
        nline.remove(i)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.

print("가로방향 ", len(nline), " 개")

for n in nline:
    print(n.pointlist)
    print('d')

#  세로 선 찾기
for j in range(0, h - 1):
    find = 0
    for i in range(0, w - 1):
        if img[j][i] != 255 and find == 0:
            find = 1
            set_line_info(nline2, i, j)
        if img[j][i] == 255 and find == 1:
            find = 0

for i in nline2[:]:
    if len(i.pointlist) < 4:
        nline2.remove(i)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.

# === 검출된 선 시각적으로 표시================================

print("세로방향 ", len(nline2), " 개")

for i in nline:
    for k in i.pointlist:
        img2[k[1]][k[0]] = 80
for i in nline:
    # img2[i.pointlist[0][1]][i.pointlist[0][0]] = [0, 255, 255]
    # img2[i.pointlist[-1][1]][i.pointlist[-1][0]] = [255, 0, 255]
    img2[i.pointlist[0][1]][i.pointlist[0][0]] = 127
    img2[i.pointlist[-1][1]][i.pointlist[-1][0]] = 127

for i in nline2:
    for k in i.pointlist:
        img3[k[1]][k[0]] = 80
for i in nline2:
    # img3[i.pointlist[0][1]][i.pointlist[0][0]] = [0, 255, 255]
    # img3[i.pointlist[-1][1]][i.pointlist[-1][0]] = [255, 0, 255]
    img3[i.pointlist[0][1]][i.pointlist[0][0]] = 127
    img3[i.pointlist[-1][1]][i.pointlist[-1][0]] = 127

print("0라인값")
# for k in nline[0].pointlist :
#    print(k[0],k[1])


cv2.imshow("vertical", img3)
cv2.imshow("horizontal", img2)
cv2.imshow("RESULT", img2 + img3)
# cv2.imshow("horizontal", img2)
# cv2.imshow("RESULT", img2 + img3)

cv2.waitKey(0)

# showResultLines(img)

"""
진행 상황

1단계
 가로로 세로운 선 찾는 정도.

2단계
 세로로도 새로운선찾기

3단계
 찾다가 두께가 너무 두꺼우면 해당 선 취소하기

4단계
 가로선 세로선 가지고 연결점 찾기
 가로선 찾다가 두꺼운점


기울기 계산하기

기울기로 가로선인지 세로선인지 판단해서
 가로선으로 계산할지 세로선으로 계산할지 판단


 어쩌면 가로세로선 구분하지말고
 시작만 가로 세로선에서 하고
 시작하고 나면 가로로 진행할지 세로로 진행할지를 기울기 진행도 가지고
  판단하기??


-가로 선을 찾을때 세로로

"""
