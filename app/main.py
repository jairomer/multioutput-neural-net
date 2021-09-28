print("Loading libraries...")
import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.utils import to_categorical
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

####################################################################################
TRAIN_TEST_SPLIT=0.7
IM_WIDTH = IM_HEIGHT = 198

script_path=os.path.split(os.path.realpath(__file__))[0]
dataset_folder_name=script_path+'/data/UTKFace'
print(dataset_folder_name)
dataset_dict = {
    'race_id' : {
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id' : {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i) for i, r in dataset_dict['race_id'].items())

####################################################################################

class Parser:
    def parse_info_from_file(path):
        """
        Parse information from a single file.
            10_1_0_20170109204859493.jpg.chip.jpg
                - 10 : Age
                - 1  : Gender
                - 0  : Race
        """
        try:
            # Get filename
            filename = os.path.split(path)[1]
            # Split the extension from a pathname. (root, ext)
            # 10_1_0_20170109204859493.jpg.chip.jpg -> 10_1_0_20170109204859493, jpg.chip.jpg
            filename = os.path.splitext(filename)[0]
            # 10_1_0_20170109204859493 -> 10, 1, 0, 20170109204859493
            age, gender, race, _ = filename.split('_')
            #print(".", end="")
            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as e:
            #print("Cannot parse:", e)
            return None, None, None

    def parse_dataset(dataset_path, ext='jpg'):
        """
        Extract information about our dataset.
        Iterate over all images and return a DataFrame with the data
        (age, gender, sex) of all files.
        """
        # Return a list of paths matching a pathname pattern.
        p = os.path.join(dataset_path, "*.%s" % ext )
        print(p)
        files = glob.glob(p)

        records = []
        for file in files:
            # Extract information
            info = Parser.parse_info_from_file(file)
            records.append(info)


        # Create dataframe from records
        df = pd.DataFrame(records)
        # Add new column
        df['file'] = files
        df.columns = ['age', 'gender', 'race', 'file']
        # Remove missing values from dataset.
        df = df.dropna()
        return df

class Plotter:
    """
    It is a recomended practice to perform some data visualization process
    on our dataset.
    """
    def plot_pie(pd_series):
        """
        Show a pie plot that contains a pie graph for a given dataframe.
        """
        labels = pd_series.value_counts().index.tolist()
        counts = pd_series.value_counts().values.tolist()
        pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
        fig = go.Figure(data=[pie_plot])
        fig.update_layout(title_text='Distribution for %s' % pd_series.name)
        fig.show()

    def plot_histogram(pd_series, nbins):
        """
        Plot a histogram from a given series and number of bins.
        """
        fig = px.histogram(pd_series, nbins=nbins)
        fig.update_layout(title_text='Distribution for %s' % pd_series.name)
        fig.show()

    def plot_accuracy(history, name):
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[name+'_output_acc'],
            name='Train'))
        fig.add_trace(go.Scatter(
            y=history.history['val_'+name+'_output_acc'],
            name='Valid'))

        fig.update_layout(height=500,
                          width=700,
                          xaxis_title='Epoch',
                          yaxis_title='Accuracy')
        fig.show()


def data_analysis(df):
    # Note: Plotly will open a browser to show the diagrams.
    #   This will not work on the docker container as it is.
    # Verify that the dataset does not overly represent a racial group.
    Plotter.plot_pie(df['race'])
    # Verify that we have the same number of men and women
    Plotter.plot_pie(df['gender'])
    # Show age distribution
    Plotter.plot_histogram(df['age'], 20)
    # Show plto pie of age
    bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
    names = ['<10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
    age_binned = pd.cut(df['age'], bins, labels=names)
    plot_pie(age_binned)


class UtkFaceDataGenerator():
    """
    In order to input our data to our Keras multi-output model, we will create a helper
    object to work as a data generator for our dataset.

    This will be done by generating batches of data, which will be used to feed our
    multioutput model with both the images and their labels.

    This step is also done instead of just loading all the dataset into the memory at
    once, which could lead to a an out of memory error.
    """
    def __init__(self, dataframe):
        self.df = dataframe

    def generate_split_indexes(self):
        # Randomly permute the current sequence.
        p = np.random.permutation(len(self.df))
        # Setup the upper bound index for the training.
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        # Set the dataset for training and testing
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        # Split again to obtain a dataset towards check validity
        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx, = train_idx[:train_up_to], train_idx[train_up_to:]

        # Convert alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()

        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        # Normalize the pixel values.
        im = np.array(im) / 255.0
        return im

    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        # Arrays to store our batched data
        images, ages, races, genders = [], [], [], []
        while True:
            for idx in image_idx:
                # A row of the dataframe is a person
                person = self.df.iloc[idx]
                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']

                im = self.preprocess_image(file)

                # Normalize age, races and genders
                ages.append(age / self.max_age)
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                genders.append(to_categorical(gender, 2))
                images.append(im)

                # yielding condition, this is a generator function.
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    # Reset batch
                    images, ages, races, genders = [], [], [], []
            if not is_training:
                break # Implementation of do-while


class UtkMultiOutputModel():
    """
    Used to generate our multioutput model.
    This CNN contains three branches:
    - age
    - sex
    - race
    Each branch contains a sequence of Convolutional Layers defined
    on the make_default_hidden_layers method.

    The default structure for our convolutional layers is based on a Conv2D
    layer with a ReLU activation, followed by a BatchNormalization layer,
    a MaxPooling and then finally a Dropout layer.

    Each of these layers is then followed by the final Dense layer.
    This step is repeated for each of the outputs we are trying to predict.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers.
        The structure used in this network is defined as:
            Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3,3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x) # Normalize input from the activation network
        x = MaxPooling2D(pool_size=(3,3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x) # Normalize input from the activation network
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x) # Normalize input from the activation network
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        return x

    def build_race_branch(self, inputs, num_races):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three conv hidden layers followed by a
        Dense output layer.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)    # Convert AxB matrix to A*Bx1 (128)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x) # Deactivate half of the hidden layer neurons during training
        x = Dense(num_races)(x)
        # Notice that the last activation layer adapts to the kind of
        # output. In this case we will use softmax to obtain a category from many.
        x = Activation("softmax", name="race_output")(x)

        return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks
        followed by the Dense output layer.
        """
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)
        # Funny, we use the grayscale as input to the CNN
        x = self.make_default_hidden_layers(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        # On this case we will use sigmoid for a binary classification.
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):
        """
        Used to build the age branch of our face recognition network.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        # On this case we want a number from a range, we will use linear.
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height, num_races):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        age_branch = self.build_age_branch(inputs)
        race_branch = self.build_race_branch(inputs, num_races=len(dataset_dict['race_id']))
        gender_branch = self.build_gender_branch(inputs, num_genders=2)

        model = Model(inputs=inputs,
                      outputs = [age_branch, race_branch, gender_branch],
                      name="face_net")
        return model

def compile_model(model, epochs = 100):
    """
    For compilation we will use a learning rate of 0.0004
    and an Adam optimizer.

    We will also use custom loss weights and a custom loss function for each feature.

    When building our optimizer, let's use a decay based on the learning rate
    divided by the number of epochs, so we will slowly be decreasing our learning
    rate over the epochs.
    """
    init_lr = 1e-4
    opt = Adam(lr=init_lr, decay=init_lr/epochs)
    # Notice that the loss, loss_weights and metrics for each of the outputs
    # of the network need to be defined.
    # Age uses mean squared error as loss function, and mae as metric for training.
    # Race uses categorical_crossentropy because we have several classes
    # Gender uses binary_crossentropy because we are assumming two genders.
    # For both race and gender we are using accuracy as training metric, because they
    # are discrete classes.
    model.compile(optimizer=opt,
                  loss={
                      'age_output' : 'mse',
                      'race_output' : 'categorical_crossentropy',
                      'gender_output' : 'binary_crossentropy'},
                  loss_weights={
                      'age_output': 4,
                      'race_output': 1.5,
                      'gender_output': 0.1},
                 metrics={
                      'age_output': 'mae',
                      'race_output': 'accuracy',
                      'gender_output': 'accuracy'})
    return model

def train_model(model, train_idx, valid_idx, data_generator, epochs=100):
    batch_size = 32
    valid_batch_size = 32
    train_gen = data_generator.generate_images(
        train_idx, is_training=True, batch_size=batch_size)
    valid_gen = data_generator.generate_images(
        valid_idx, is_training=True, batch_size=valid_batch_size)

    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    history = model.fit_generator(train_gen,
                        steps_per_epoch = len(train_idx)//batch_size,
                        epochs = epochs,
                        callbacks = callbacks,
                        validation_data = valid_gen,
                        validation_steps= len(valid_idx)//valid_batch_size,
                        verbose = 1)

    Plotter.plot_accuracy(history, "race")
    Plotter.plot_accuracy(history, "gender")
    Plotter.plot_accuracy(history, "age")

    return model

def test_model(model, test_idx):
    test_batch_size = 128
    test_generator = data_generator.generate_images(
        test_idx, is_training=False, batch_size=test_batch_size)

    age_pred, race_pred, gender_pred = model.predict_generator(
        test_generator, steps=len(test_idx)//test_batch_size)




def main():
    df = Parser.parse_dataset(dataset_folder_name)
    print(df.head())
    #data_analysis(df)
    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
    model = UtkMultiOutputModel().assemble_full_model(
        IM_WIDTH, IM_HEIGHT, num_races=len(dataset_dict['race_alias']))
    model = compile_model(model)
    trained_model = train_model(model, train_idx, valid_idx, data_generator)

if __name__ == '__main__':
    main()

