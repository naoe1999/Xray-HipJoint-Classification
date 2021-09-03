import os
import cv2
import numpy as np
import sklearn.metrics as skm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images(image_file_list):
    img_list = []
    for f in image_file_list:
        img = image.img_to_array(image.load_img(f))
        img = img / 255.
        img_list.append(img)
    return np.array(img_list)


IMG_DIR = './dataset/train'
CAM_SAVE_DIR = './cam'

BINARY = True
# CATEGORY = ['Normal', 'Mild', 'Moderate', 'Severe']
CATEGORY = ['Abnormal', 'Normal']
ABNORMAL = 0


if __name__ == '__main__':

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    valid_generator = datagen.flow_from_directory(
        './dataset/train', target_size=(224, 224), class_mode='categorical', shuffle=False,
        subset='validation'
    )

    cls_dict = valid_generator.class_indices
    cls_name_dict = {v: k for k, v in cls_dict.items()}
    print(cls_name_dict)

    imgs, labels = valid_generator.next()
    labels = np.argmax(labels, axis=1)
    img_shape = imgs.shape[1:3]
    print(imgs.shape, labels.shape)

    # load model
    model = load_model('./trained_model.h5')

    # generate CAM model
    gmodel = Model(inputs=model.inputs, outputs=[model.get_layer('batch_normalization_2').output, model.output])
    gmodel.summary()

    # predict for all images
    with tf.GradientTape() as tape:
        (conv_feats, preds) = gmodel(imgs)
        if BINARY:
            ys = preds[:, ABNORMAL]
        else:
            ys = tf.reduce_max(preds, axis=1)

    probs = tf.nn.softmax(preds, axis=1)

    # calculate CAM
    grads = tape.gradient(ys, conv_feats)

    weights = tf.reduce_mean(grads, axis=(1, 2))
    weights = tf.reshape(weights, [tf.shape(weights)[0], 1, 1, -1])
    cams = tf.reduce_sum(tf.multiply(weights, conv_feats), axis=-1)
    cams = tf.maximum(cams, 0)

    cams = cams.numpy()
    preds = preds.numpy()
    probs = probs.numpy()
    print(cams.shape, preds.shape, probs.shape)

    accuracy = (np.argmax(preds, axis=1) == labels).mean()
    print(accuracy * 100)

    # make cam images
    for i, (img, label, cam, pred, prob) in enumerate(zip(imgs, labels, cams, preds, probs)):
        # generate heatmap
        heatmap = cv2.resize(cam, img_shape[::-1])
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
        heatmap = (heatmap * 255).astype('uint8')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # merge with source image
        img = (img * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # get prediction score
        pred_cls = np.argmax(pred)
        if BINARY:
            pred_score = prob[ABNORMAL]
        else:
            pred_score = prob[pred_cls]

        out_filename = f'cam{i:02d}_{cls_name_dict[label]}_{cls_name_dict[pred_cls]}_{pred_score:.2f}.png'
        out_filepath = os.path.join(CAM_SAVE_DIR, out_filename)

        # create dir if not exists
        if not os.path.exists(CAM_SAVE_DIR):
            os.makedirs(CAM_SAVE_DIR)

        # write image file
        cv2.imwrite(out_filepath, output)

    # confusion matrix
    y_preds = np.argmax(preds, axis=1)
    cf_mat = skm.confusion_matrix(labels, y_preds)

    lbl = list(cls_name_dict.values())
    df_cm = pd.DataFrame(cf_mat, index=lbl, columns=lbl)
    df_cm = df_cm[CATEGORY]
    df_cm = df_cm.reindex(CATEGORY)

    sns.heatmap(df_cm, annot=True, cmap='Blues')
    plt.show()

    print('done')

