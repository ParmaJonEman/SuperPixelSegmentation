import cv2
import numpy as np
import argparse


def slic_segmentation(image, labels):
    # Calculate average color of a superpixel
    reshaped = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    labels_flattened = np.reshape(labels, -1)
    unique_labels = np.unique(labels_flattened)
    mask = np.zeros(reshaped.shape)
    for i in unique_labels:
        loc = np.where(labels_flattened == i)[0]
        mask[loc, :] = np.mean(reshaped[loc, :], axis=0)
    return np.reshape(mask, [image.shape[0], image.shape[1], image.shape[2]]).astype('uint8')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="*")
    ap.add_argument("-a", "--algorithm", required=False,
                    help="Superpizel Algorithm to use (SLIC, SLICO, or MSLIC, default SLIC)")
    ap.add_argument("-r", "--ruler", required=False, help="Ruler value (default 10.0)")
    ap.add_argument("-s", "--size", required=False, help="Superpixel size value (default 10)")

    args = vars(ap.parse_args())
    image = args['image'][0]
    print(args)
    if args['ruler']:
        ruler = float(args['ruler'])
    else:
        ruler = 10.0

    if args['size']:
        size = int(args['size'])
    else:
        size = 10

    if args['algorithm'] == "SLIC":
        algorithm = cv2.ximgproc.SLIC
    elif args['algorithm'] == "SLICO":
        algorithm = cv2.ximgproc.SLICO
    elif args['algorithm'] == "MSLIC":
        algorithm = cv2.ximgproc.MSLIC
    else:
        algorithm = cv2.ximgproc.SLIC

    return image, ruler, size, algorithm


if __name__ == '__main__':
    image, ruler, size, algorithm = parse_args()

    src = cv2.imread(image)  # read image
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)  # convert to LAB

    slic = cv2.ximgproc.createSuperpixelSLIC(src_lab, algorithm=algorithm,
                                        region_size=size, ruler=ruler)
    slic.iterate()

    labels = slic.getLabels()
    segmented_image = slic_segmentation(src, labels)

    mask_slic = slic.getLabelContourMask()
    mask_inv_slic = cv2.bitwise_not(mask_slic)

    segments = cv2.bitwise_and(segmented_image, segmented_image, mask=mask_inv_slic)
    cv2.imshow("slic", src)
    cv2.waitKey()
    cv2.imshow("slic", segmented_image)
    cv2.waitKey()
    cv2.imshow("slic", segments)
    cv2.waitKey()
