import os
import cv2
import numpy as np
from kmeans import kmeans

if __name__ == "__main__":
    img_name = "12"
    img_path = os.path.join("imgs", img_name + ".jpg")
    img = cv2.imread(img_path) 
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kmeans_cluster = kmeans(k=3, max_iters=100, color_space="YCbCr", channels=[1])
    result = kmeans_cluster.clustering(img)
    cv2.imshow("Input image", img)
    cv2.imshow("Extracted number", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_path = os.path.join("imgs", f"{img_name}_result.jpg")
    cv2.imwrite(img_path, result)
