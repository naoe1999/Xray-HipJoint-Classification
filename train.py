
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, \
     GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy


NUM_CLASS = 2


if __name__ == '__main__':

    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True, rotation_range=5.0, zoom_range=0.1, brightness_range=[0.9, 1.1],
        validation_split=0.1
    )

    train_generator = datagen.flow_from_directory(
        './dataset/train', target_size=(224, 224), class_mode='categorical', shuffle=True, batch_size=8,
        subset='training'#, save_to_dir='./data/train', save_prefix='train'
    )
    valid_generator = datagen.flow_from_directory(
        './dataset/train', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=8,
        subset='validation'#, save_to_dir='./data/valid', save_prefix='valid'
    )

    input_tensor = Input(shape=(224, 224, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    vgg16.trainable = False

    model = Sequential()
    model.add(vgg16)
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(NUM_CLASS, activation=None))

    loss = CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=RMSprop(1e-4), loss=loss, metrics=['accuracy'])
    model.summary()

    model.fit(train_generator, epochs=100, validation_data=valid_generator)

    model.evaluate(valid_generator)
    model.save('./trained_model.h5')

