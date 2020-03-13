import os
import cv2
import matplotlib.pyplot as plt

mask = cv2.imread(os.path.join('..', 'S1', 'Seq1', 'BGmasks', 'mask.jpg'), cv2.IMREAD_GRAYSCALE)
plt.hist(mask.reshape(-1))
plt.show()
