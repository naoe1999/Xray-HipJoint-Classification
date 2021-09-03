import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


DEBUG = False
CATEGORY = ['Normal', 'Mild', 'Moderate', 'Severe']
SRCDIR = '../Learning'
IMGSIZE = (224, 224)


def get_image_data_list():
    image_data_list = []
    image_array = []

    for cls in CATEGORY:
        img_files = sorted(os.listdir(os.path.join(SRCDIR, cls)))

        for imgf in img_files:
            img = cv2.imread(os.path.join(SRCDIR, cls, imgf))
            mean = np.mean(img)
            med = np.median(img)
            std = np.std(img)
            height, width = img.shape[0:2]

            src = 'SNU' if 'SNU' in imgf else 'IAN'
            cls_regroup = 'Normal' if (cls == 'Normal' or cls == 'Mild') else 'Abnormal'

            # if min(width, height) < 200:
            #     continue
            # if src != 'SNU':
            #     continue

            image_data_list.append((width, height, med, mean, std, cls_regroup, src))

            img_resize = cv2.resize(img, IMGSIZE)
            img_norm = normalize(img_resize)
            image_array.append(img_norm)

    image_array = np.array(image_array)
    return image_data_list, image_array


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def plot_pair(image_data_list):
    df = pd.DataFrame(data=image_data_list, columns=['width', 'height', 'median', 'mean', 'std', 'class', 'from'])
    if DEBUG:
        print(df.head(10))

    g = sns.PairGrid(df, hue='class')
    g.map_diag(sns.histplot, element='step')
    g.map_offdiag(sns.scatterplot, size=df['from'])
    g.add_legend(title="", adjust_subtitles=True)
    plt.show()


def plot_tsne(image_array, image_data_list):
    imgs = image_array.reshape((image_array.shape[0], -1))
    labels = [data[5] for data in image_data_list]
    # labels = [data[6] for data in image_data_list]

    tsne = TSNE(n_components=2, n_iter=5000, random_state=0)
    res = tsne.fit_transform(imgs)

    if DEBUG:
        print(imgs.shape)
        print(len(labels))
        print(res.shape)

    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=labels)
    plt.show()


def main():
    image_data_list, image_array = get_image_data_list()
    print(f'{len(image_data_list)} images are loaded.')
    print(image_array.shape)

    plot_pair(image_data_list)

    plot_tsne(image_array, image_data_list)


# main
if __name__ == '__main__':
    main()
