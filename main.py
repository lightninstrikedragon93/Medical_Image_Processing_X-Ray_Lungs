import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_contrast_measure(lung_segment):
    #standard deviation of pixel intensities
    std_dev = np.std(lung_segment)
    return std_dev

#load image
image = cv2.imread(input('Enter image path'), cv2.IMREAD_GRAYSCALE)

#thresholdig binary
image_tresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 2)
plt.imshow(image_tresh)
plt.show()

# noise removal 1
image_tresh = cv2.morphologyEx(image_tresh, cv2.MORPH_OPEN, kernel = np.ones ((2, 2), np.uint8))
plt.imshow(image_tresh)
plt.show()

# find contours
contours, _ = cv2.findContours(image_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# create mask for the contour
contour_mask = np.zeros_like(image_tresh)

# draw contours on the mask
cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.LINE_4)
plt.imshow(contour_mask)
plt.show()

# apply the contour mask to the original image
image_tresh = cv2.bitwise_not(image_tresh, image_tresh, mask=contour_mask)
plt.imshow(image_tresh)
plt.show()

#noise removal 2

# image_tresh = cv2.dilate(image_tresh, kernel = np.ones([5,5]))
image_tresh = cv2.erode(image_tresh, kernel = np.ones([5,5]))
image_tresh = cv2.dilate(image_tresh, kernel = np.ones([5,5]))
image_tresh = cv2.erode(image_tresh, kernel = np.ones([5,5]))
image_tresh = cv2.morphologyEx(image_tresh, cv2.MORPH_OPEN, kernel=np.ones([10,10]))

plt.imshow(image_tresh)
plt.show()

processed_masks = []
# connected component analysis
num_labels, labels = cv2.connectedComponents(image_tresh.astype(np.uint8))
good_labels = []

image_height, image_width = image.shape[:2]
center_x = image_width // 2 + 50

#column indices for halfway through the left and right of the center
left_halfway_column = center_x // 2
right_halfway_column = center_x + (image_width - center_x) // 2
    
for label in range(1, num_labels):
    #bounding box of current label
    region_mask = (labels == label).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(region_mask)

    if np.any(image_tresh[:, left_halfway_column] == 255) or np.any(image_tresh[:, right_halfway_column] == 255):
    # if 80 < h < 250  and 70 < w < 130  and  x + w < 250 and y + h < 250 :
        good_labels.append(label)
    
#create mask containing only good regions    
mask = np.zeros_like(labels)

for label in good_labels:
    mask[labels == label] = label

processed_masks.append(mask)

#filling internal holes
p_final_masks = []
for im_th in processed_masks:

    im_th = im_th.astype(np.uint8)

    contours, hierarchy = cv2.findContours(im_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    internal_contours = np.zeros_like(im_th)
    external_contours = np.zeros_like(im_th)
    
    for i, contour in enumerate(contours):
        #check if contour is external
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contour)
            if area > 244.0:
                cv2.drawContours(external_contours, contours, i, 255, -1)
    
    #dilate external contours mask
    external_contours = cv2.dilate(external_contours, kernel=np.ones((7, 7), dtype=np.uint8))
    
    p_final_masks.append(external_contours)

for mask in p_final_masks:
    plt.imshow(mask, cmap="gray")
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
plt.show()

final_masks = []
for img in p_final_masks:
    img = cv2.bitwise_not(img.astype(np.uint8))
    
    img = cv2.erode(img, kernel=np.ones((5,5)))
    
    img = cv2.bitwise_not(img)
    
    img = cv2.dilate(img, kernel=np.ones((5,5)))
    
    img = cv2.erode(img, kernel=np.ones((1,1)))
    
    final_masks.append(img)

for mask in final_masks:
    plt.imshow(mask, cmap="gray")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
plt.show()

size = cv2.contourArea(contours[1])
size = size * 0.1 ** 2
print('Size:', size)
#calculate contrast measure
image_contrast_measure = calculate_contrast_measure(image_tresh)

#sum of the white pixels
images_white_pixels = np.sum(image_tresh == 255)

print('contrast measure: ', image_contrast_measure)
print('sum of the white pixels:', images_white_pixels)

if 110 <= image_contrast_measure <= 130 and 15000 <= images_white_pixels <= 65000 and size >= 65:
    print('normal')
elif 70 <= image_contrast_measure <= 110 and images_white_pixels <= 30000 and size < 65:
        print('unhealthy')
else:
    print('no defined')

plt.show()