import cv2
import something


image_path = "C:/Users/USER/workspace/palm/images/sample1.png"

print("a")
image = cv2.imread(image_path)
print("b")
image = something.get_palm_coordinate(image)
print("c")
cv2.imwrite("palm_only1.png",image)
print("d")
cv2.imshow("palm_only1.png",image)
print("e")
cv2.waitKey(0)
print("f")
