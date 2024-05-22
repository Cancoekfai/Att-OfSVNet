import cv2

img = cv2.imread('CEDAR/inputs/forgeries_1_1.png')
cv2.imwrite('CEDAR/inputs/forgeries_1_1_inv.png', 255 - img)
img = cv2.imread('BHSig-B/inputs/forgeries_1_1.png')
cv2.imwrite('BHSig-B/inputs/forgeries_1_1_inv.png', 255 - img)
cv2.imwrite('BHSig-B/inputs/forgeries_1_1_copy.png', img)
img = cv2.imread('BHSig-H/inputs/forgeries_1_1.png')
cv2.imwrite('BHSig-H/inputs/forgeries_1_1_inv.png', 255 - img)
cv2.imwrite('BHSig-H/inputs/forgeries_1_1_copy.png', img)
img = cv2.imread('CNSig/inputs/forgeries_11_2.png')
cv2.imwrite('CNSig/inputs/forgeries_11_2_inv.png', 255 - img)