import cv2


class hline:
    def __init__(self):
        self.pointlist = []
        return

    def addLine(self, w, h):
        self.pointlist.append([w, h])

    def isContinue(self, w, h):
        if abs(self.pointlist[-1][1] - h) < 3 \
                and abs(self.pointlist[-1][0] - w) < 3:
            return True
        return False


def setLineInfo(nline, img, w, h):
    find = False
    for n in nline:
        if n.isContinue(w, h) == True:
            find = True
            n.addLine(w, h)
    if find == False:
        hl = hline()
        nline.append(hl)
        hl.addLine(w, h)
    return find


def showResultLines(img):
    cv2.imshow("image", img)
    k = cv2.waitKey(0)  # 키보드 눌림 대기
    if k == 27:  # ESC키
        cv2.destroyAllWindows()
    elif k == ord('s'):  # 저장하기 버튼
        cv2.imwrite("./opencv/test2.png", img)
        cv2.destroyAllWindows()
    return
