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

    def preprocess_image(self, image_path):
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
        while is_training:
            for idx in image_idx:
                # A row of the dataframe is a person
                person = self.df.iloc[idx]
                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']

                im = self.preprocess_image(file)

                # Normalize age, races and genders
                ages.append(age / sefl.max_age)
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                genders.append(to_categorical(genders, len(dataset_dict['gender_id'])))
                images.append(im)

                # yielding condition, this is a generator function.
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    # Reset batch
                    images, ages, races, genders = [], [], [], []


def main():
    df = Parser.parse_dataset(dataset_folder_name)
    print(df.head())
    #data_analysis(df)
    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

if __name__ == '__main__':
    main()

