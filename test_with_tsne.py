import cv2
import numpy as np
import sklearn.metrics as skm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_DIR = './dataset/train'

# BINARY = False
# CATEGORY = ['Normal', 'Mild', 'Moderate', 'Severe']

BINARY = True
CATEGORY = ['Abnormal', 'Normal']
ABNORMAL = 0


def load_images(image_file_list):
    img_list = []
    for f in image_file_list:
        img = image.img_to_array(image.load_img(f))
        img = img / 255.
        img_list.append(img)
    return np.array(img_list)


def plot_tsne(data_array, labels, title=None):
    data = data_array.reshape((data_array.shape[0], -1))

    tsne = TSNE(n_components=2, n_iter=1000, random_state=0)
    res = tsne.fit_transform(data)

    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=labels)
    if title:
        plt.title(title)
    plt.show()


if __name__ == '__main__':

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    train_generator = datagen.flow_from_directory(
        './dataset/train', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=32,
        subset='training'
    )
    valid_generator = datagen.flow_from_directory(
        './dataset/train', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=32,
        subset='validation'
    )

    cls_dict = valid_generator.class_indices
    cls_name_dict = {v: k for k, v in cls_dict.items()}
    print(cls_name_dict)

    labels_tr = train_generator.labels
    labels_val = valid_generator.labels

    imgs_tr = []
    for imgf in train_generator.filepaths:
        img = cv2.imread(imgf)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs_tr.append(img)
    imgs_tr = np.asarray(imgs_tr)
    print(imgs_tr.shape)

    imgs_val = []
    for imgf in valid_generator.filepaths:
        img = cv2.imread(imgf)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs_val.append(img)
    imgs_val = np.asarray(imgs_val)
    print(imgs_val.shape)

    # load model
    model = load_model('./trained_model.h5')
    model.summary()

    # evaluation
    print('training data:')
    model.evaluate(train_generator)

    print('validation data:')
    model.evaluate(valid_generator)

    # make a model to extract features
    new_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer('batch_normalization_2').output,
            model.get_layer('global_average_pooling2d').output,
            model.output
        ]
    )

    # for training data
    # prediction
    feats, gaps, preds = new_model.predict(train_generator)
    print('training:', feats.shape, gaps.shape, preds.shape)

    # confusion matrix
    y_preds = np.argmax(preds, axis=1)
    cf_mat = skm.confusion_matrix(labels_tr, y_preds)

    lbl = list(cls_name_dict.values())
    df_cm = pd.DataFrame(cf_mat, index=lbl, columns=lbl)
    df_cm = df_cm[CATEGORY]
    df_cm = df_cm.reindex(CATEGORY)

    sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.show()

    # t-SNE
    plot_tsne(imgs_tr, labels_tr, 't-SNE for training images')
    plot_tsne(feats, labels_tr, 't-SNE for training feature matrix')
    plot_tsne(gaps, labels_tr, 't-SNE for training GAP vectors')

    # for validation data
    # prediction
    feats_v, gaps_v, preds_v = new_model.predict(valid_generator)
    print('validation:', feats_v.shape, gaps_v.shape, preds_v.shape)

    # merge and plot
    imgs = np.vstack([imgs_tr, imgs_val])
    labels = np.hstack([labels_tr, labels_val])
    print('total:', imgs.shape, len(labels))

    feats = np.vstack([feats, feats_v])
    gaps = np.vstack([gaps, gaps_v])
    preds = np.vstack([preds, preds_v])
    print('total:', feats.shape, gaps.shape, preds.shape)

    # confusion matrix
    y_preds = np.argmax(preds, axis=1)
    cf_mat = skm.confusion_matrix(labels, y_preds)
    df_cm = pd.DataFrame(cf_mat, index=lbl, columns=lbl)
    df_cm = df_cm[CATEGORY]
    df_cm = df_cm.reindex(CATEGORY)
    sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.show()

    # t-SNE
    plot_tsne(imgs, labels, 't-SNE for all images')
    plot_tsne(feats, labels, 't-SNE for all feature matrix')
    plot_tsne(gaps, labels, 't-SNE for all GAP vectors')

