import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Unet import Unet
from ResUnet import ResUnet
from keras.metrics import MeanIoU, Recall, Precision
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

IMG_PATH = 'rgb/*'
MSK_PATH = 'mask/*'

BATCH_SIZE = 1
EPOCHS = 80

SIZE = 512
IMG_CHANNELS = 3
MSK_CHANNELS = 1
TRAIN_SIZE = 0.8 
TEST_SIZE = 0.2
RANDOM_STATE = 42
PREFETCH = 4
LEARNING_RATE = 1e-4

# reads an RGB image
# output shape = (IMG_SIZE, IMG_SIZE, 3)
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (SIZE, SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

# reads grayscale mask
# output shape = (IMG_SIZE, IMG_SIZE, 1)
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (SIZE, SIZE))
    x = x/255.0
    x = x.astype(np.float32)
    # (IMG_SIZE, IMG_SIZE) -> (IMG_SIZE, IMG_SIZE, 1)
    x = np.expand_dims(x, axis=-1) # -1 -> last axis
    return x

def load_dataset(dataset_path):
    images = sorted(glob(os.path.join(dataset_path, IMG_PATH)))
    masks = sorted(glob(os.path.join(dataset_path, MSK_PATH)))
    train_x, test_x = train_test_split(images, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_y, test_y = train_test_split(masks, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return (train_x, train_y), (test_x, test_y)

def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()
        x = read_image(image_path)
        y = read_mask(mask_path)
        return x, y

    # tf.numpy_function takes: function name, params, output data type
    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([SIZE, SIZE, IMG_CHANNELS])
    mask.set_shape([SIZE, SIZE, MSK_CHANNELS])
    return image, mask

# takes a list of images/mask path
def tf_dataset(images, masks, batch=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=2*len(dataset))
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(PREFETCH)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data', help='Path to the training set directory.', required=True)
    parser.add_argument('--model', help='Directory for saving models.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # TRAIN    
    # load the dataset
    (train_x, train_y), (test_x, test_y) = load_dataset(args.data)
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    test_dataset = tf_dataset(test_x, test_y, batch=BATCH_SIZE)

    # build the model
    # model = Unet((SIZE, SIZE, IMG_CHANNELS)).build_net()
    model = ResUnet(SIZE, IMG_CHANNELS).build_net()

    model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    metrics=[
        MeanIoU(num_classes=2),
        Recall(),
        Precision()
        ]
    )

    # model.summary()

    callbacks = [
        ModelCheckpoint(args.model, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10)
    ]
    # CSVLogger(CSV_PATH),

    train_steps = len(train_x)//BATCH_SIZE
    if len(train_x) % BATCH_SIZE != 0:
        train_steps += 1

    test_steps = len(test_x)//BATCH_SIZE
    if len(test_x) % BATCH_SIZE != 0:
        test_steps += 1

    # train the model
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()