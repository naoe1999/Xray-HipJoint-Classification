import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


SRCDIR = '../Learning'
DATADIR = '../dataset/train'

DATASET = ['Normal', 'Mild', 'Moderate', 'Severe']

DATASET_REGROUP = {
    'Normal': ['Normal'],
    'Abnormal': ['Mild', 'Moderate', 'Severe']
}

IMGSIZE = (224, 224)


def get_image(filepath, target_wh=None, return_info=False):
    img = cv2.imread(filepath)

    # (width, height, min, max, mean, std)
    info = (img.shape[1], img.shape[0], np.min(img), np.max(img), np.mean(img), np.std(img))

    # crop
    h, w, _ = img.shape
    if h > w:
        img = img[(h - w) // 2: (h - w) // 2 + w, :, :]
    elif h < w:
        img = img[:, (w - h) // 2: (w - h) // 2 + h, :]

    # resize
    if target_wh:
        img = cv2.resize(img, target_wh)

    # brightness and contrast adjustment
    imin = np.min(img)
    imax = np.max(img)
    img = ((img - imin) / (imax - imin) * 255).astype(np.int)

    if return_info:
        return img, info
    else:
        return img


def build_dataset(dataset, target_wh, show_distribution=True, balancing=False):

    # preprocess source image files
    img_list_dict = {}
    infos = []

    if type(dataset) is list:

        for cls in dataset:
            imgfiles = sorted(os.listdir(os.path.join(SRCDIR, cls)))
            print(cls, len(imgfiles))

            img_list = []
            for imgf in imgfiles:
                # filter by source
                if 'SNU' not in imgf:
                    continue

                img, info = get_image(os.path.join(SRCDIR, cls, imgf), target_wh=target_wh, return_info=True)

                # filter by size
                # if min(info[0:2]) < 200:
                #     continue

                img_list.append(img)
                infos.append(info)

            img_list_dict[cls] = img_list
            print(cls, len(img_list), 'after filtering')

    elif type(dataset) is dict:

        for grp, sub_classes in dataset.items():

            img_list = []
            for sub_cls in sub_classes:
                imgfiles = sorted(os.listdir(os.path.join(SRCDIR, sub_cls)))
                print(grp, sub_cls, len(imgfiles))

                for imgf in imgfiles:
                    # filter by source
                    if 'SNU' not in imgf:
                        continue

                    img, info = get_image(os.path.join(SRCDIR, sub_cls, imgf), target_wh=target_wh, return_info=True)

                    # filter by size
                    # if min(info[0:2]) < 200:
                    #     continue

                    img_list.append(img)
                    infos.append(info)

            random.shuffle(img_list)
            img_list_dict[grp] = img_list
            print(grp, len(img_list), 'after filtering and merging')

    else:
        print('cannot handle dataset type:', type(dataset))
        return

    # plot distribution of source data
    if show_distribution:
        # size (w, h) information of source images
        ws = np.array([info[0] for info in infos])
        hs = np.array([info[1] for info in infos])
        print('width range:', np.min(ws), '~', np.max(ws), ' mean:', np.mean(ws))
        print('height range:', np.min(hs), '~', np.max(hs), ' mean:', np.mean(hs))
        plt.hist(np.stack([ws, hs], axis=1), bins=50, histtype='bar')
        plt.show()

        # image value information of source images
        mmms = np.array([info[2:] for info in infos])
        print('min value range:', np.min(mmms[:, 0]), '~', np.max(mmms[:, 0]), ' mean:', np.mean(mmms[:, 0]))
        print('max value range:', np.min(mmms[:, 1]), '~', np.max(mmms[:, 1]), ' mean:', np.mean(mmms[:, 1]))
        print('mean value range:', np.min(mmms[:, 2]), '~', np.max(mmms[:, 2]), ' mean:', np.mean(mmms[:, 2]))
        plt.hist(mmms, bins=50, histtype='bar')
        plt.show()

    # class balancing
    if balancing:
        num_per_cls = min([len(img_list_dict[c]) for c in img_list_dict])
        img_list_dict = {c: random.sample(img_list, num_per_cls) for c, img_list in img_list_dict.items()}

    # copy to target dirs
    os.makedirs(DATADIR, exist_ok=True)
    for cls in img_list_dict:
        cls_dir = os.path.join(DATADIR, cls)
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)

        img_list = img_list_dict[cls]

        for i, img in enumerate(img_list):
            imgfile = f'{cls}_{i:03d}.jpg'
            imgfile = os.path.join(cls_dir, imgfile)
            cv2.imwrite(imgfile, img)


if __name__ == '__main__':

    build_dataset(DATASET_REGROUP, IMGSIZE, show_distribution=True, balancing=True)

