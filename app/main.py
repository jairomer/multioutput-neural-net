import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
        #df.columns = ['age', 'gender', 'race', 'file']
        # Remove missing values from dataset.
        df = df.dropna()
        return df

df = Parser.parse_dataset(dataset_folder_name)
print(df.head())


