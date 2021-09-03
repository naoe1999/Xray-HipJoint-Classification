import os
import cv2
import csv
import numpy as np


DATASET = ['Normal', 'Mild', 'Moderate', 'Severe']
SRCDIR = './Learning'


# entity data
# ( category, source['SNU', 'IAN'], p_no, p_name, side['L', 'R'], size(w, h), min, max, mean, std )


if __name__ == '__main__':

    data_list = []

    for category in DATASET:
        filenames = sorted(os.listdir(os.path.join(SRCDIR, category)))

        for fe in filenames:
            f = os.path.splitext(fe)[0]
            s = f.split('_')

            source = 'IAN'

            if len(s) == 4:
                assert s[1] == 'SNU'
                source = 'SNU'
                del(s[1])

            [p_no, p_name, side] = s

            img = cv2.imread(os.path.join(SRCDIR, category, fe))
            size = img.shape[:2]

            imin, imax, imean, istd = map(lambda fn: fn(img), [np.min, np.max, np.mean, np.std])

            data_list.append((category, source, p_no, p_name, side, size[1], size[0], imin, imax, imean, istd))

    print(len(data_list))

    with open('../data.csv', 'w') as fp:
        cw = csv.writer(fp)

        for data in data_list:
            cw.writerow(data)

    print('done')

