import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


import sklearn
from sklearn.model_selection import train_test_split


## directory path where the data is stored
data_dir = "./data/"

## csv column names
colnames = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]


def get_image_angle_pair(df, image_type, correction):
    """
    Returns:
        dataframe which has [image_path, steering_angle] columns.
        `correction` is used for left or right side images.
    """
    df_new = df[[image_type]]
    df_new["steering_angle"] = df["steering_angle"] + correction
    df_new.columns = ["image_path", "steering_angle"]
    return df_new


def drop_angle(df, angle, drop_frac):
    """
    Remove some recored specified by angle argument
    """
    df = df.drop(df[df.steering_angle == angle].sample(frac=drop_frac).index)
    return df.reset_index(drop=True)


def load_log_files():
    """
    In order to compare the results of different training data
    I collected multiple `driving_log.csv` files

    Here, the following directory structure is assumed.
    ------------------------------------
    {data_dir}/drive_1/driving_log.csv
    {data_dir}/drive_1/IMG/...

    {data_dir}/drive_2/driving_log.csv
    {data_dir}/drive_2/IMG/...

    ...
    -------------------------------------

    """

    ## recoreded image are stored in different directories
    log_normal = data_dir+ "drive_1/" + "driving_log.csv"
    log_counter_clockwise = data_dir+ "drive_2/" + "driving_log.csv"
    log_add =  data_dir+ "drive_3/" + "driving_log.csv"
    log_left_side =  data_dir+ "drive_left_side/" + "driving_log.csv"
    log_right_side =  data_dir+ "drive_right_side/" + "driving_log.csv"

    ## load log data into pandas dataframes
    df_normal  = pd.read_csv(log_normal, header=None, names=colnames)
    df_counter_clockwise  = pd.read_csv(log_counter_clockwise, header=None, names=colnames)
    df_add  = pd.read_csv(log_add, header=None, names=colnames)
    df_left_side  = pd.read_csv(log_left_side, header=None, names=colnames)
    df_right_side  = pd.read_csv(log_right_side, header=None, names=colnames)

    ## select the only recovering scene.
    df_left_side  = df_left_side[df_left_side.steering_angle > 0.25]
    df_right_side  = df_right_side[df_right_side.steering_angle < -0.25]

    ## here, all data are conbimed.
    df_log = pd.concat([
        df_normal,
        df_counter_clockwise,
        df_add,
        df_left_side,
        df_right_side,
    ], axis=0)

    ## Downsamples straight dirve image
    df_log = drop_angle(df_log, 0.0, 0.33)

    ## include multi camera image with correction
    df_log = pd.concat([
        get_image_angle_pair(df_log, 'center_image', 0),
        get_image_angle_pair(df_log, 'left_image', 0.25),
        get_image_angle_pair(df_log, 'right_image', -0.25),
    ], axis=0)

    return df_log.reset_index(drop=True)


def generator(df_samples, batch_size=32):
    num_samples = df_samples.shape[0]
    while 1: # Loop forever so the generator never terminates

        ## shuffle dataframe
        ## https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        df_sample = df_samples.sample(frac=1).reset_index(drop=True)

        for offset in range(0, num_samples, batch_size):
            df_batch_samples = df_samples.iloc[offset:offset+batch_size, :]

            images = []
            angles = []
            for i, batch_sample in df_batch_samples.iterrows():
                image = get_image(batch_sample, data_dir)
                angle = float(batch_sample["steering_angle"])

                ## flip image and angle in a probability 0.5
                if np.random.rand() > 0.5:
                    image = np.fliplr(image)
                    angle = -angle
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def get_image(row, data_dir):
    drive_n = row["image_path"].split('/')[-3]
    fname = row["image_path"].split('/')[-1]
    image_path = data_dir + drive_n + '/IMG/'+ fname
    image = cv2.imread(image_path)
    assert image is not None, "image should not be None"
    return image

df = load_log_files()
df_train_samples, df_validation_samples = train_test_split(df, test_size=0.2, random_state=11)

batch_size = 32
train_generator = generator(df_train_samples, batch_size=batch_size)
validation_generator = generator(df_validation_samples, batch_size=batch_size)

## check count of data
print(df.shape)
#df.steering_angle.hist(bins=30)


#### mode definition
import keras
print(keras.__version__)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def get_nvidia():
    """
    NVIDIA's architecture including dropout layers are used.

    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    IMAGE_SHAPE =  (160, 320, 3)
    dropout_ratio = 0.2
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)),  input_shape=IMAGE_SHAPE))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

## train model
model = get_nvidia()
opt = keras.optimizers.Adam(0.001)
model.compile(loss="mse", optimizer=opt)
hist = model.fit_generator(train_generator, samples_per_epoch=df_train_samples.shape[0],\
                           validation_data=validation_generator, nb_val_samples=df_validation_samples.shape[0],\
                           nb_epoch=15, verbose = 2)

## output result
pd.DataFrame(hist.history).plot()
model.save("model.h5")
