# statoil-iceberg-classifier-challenge
statoil-iceberg-classifier-challenge
Statoil/C-CORE Iceberg Classifier Challenge
Identifies if a remotely sensed target is a ship or iceberg.
Project: Build a Iceberg Classifier using Deep Learning
Kaggle Statoil Iceberg Classifier Challenge
Step 1: Import Libraries
In [5]:

import numpy as np
import pandas as pd
​
from matplotlib import pyplot as plt
%matplotlib inline
Step 2: Data
In [6]:

#Change to your directory here
# train.json and test.json are expected in this folder
DATA_PATH = './data/'
TRAIN_PATH = DATA_PATH + "train.json"
TEST_PATH  = DATA_PATH + "test.json"
​
train_df = pd.read_json(TRAIN_PATH)
train_df.info()
train_df.head()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1604 entries, 0 to 1603
Data columns (total 5 columns):
band_1        1604 non-null object
band_2        1604 non-null object
id            1604 non-null object
inc_angle     1604 non-null object
is_iceberg    1604 non-null int64
dtypes: int64(1), object(4)
memory usage: 62.7+ KB
Out[6]:
band_1	band_2	id	inc_angle	is_iceberg
0	[-27.878360999999998, -27.15416, -28.668615, -...	[-27.154118, -29.537888, -31.0306, -32.190483,...	dfd5f913	43.9239	0
1	[-12.242375, -14.920304999999999, -14.920363, ...	[-31.506321, -27.984554, -26.645678, -23.76760...	e25388fd	38.1562	0
2	[-24.603676, -24.603714, -24.871029, -23.15277...	[-24.870956, -24.092632, -20.653963, -19.41104...	58b2aaa0	45.2859	1
3	[-22.454607, -23.082819, -23.998013, -23.99805...	[-27.889421, -27.519794, -27.165262, -29.10350...	4cfc3a18	43.8306	0
4	[-26.006956, -23.164886, -23.164886, -26.89116...	[-27.206915, -30.259186, -30.259186, -23.16495...	271f93f4	35.6256	0
In [7]:

def clean_dataframe(df):
    df['inc_angle'] = df['inc_angle'].replace("na", np.nan, inplace=False).astype(float)
    mean_inc_angle = df['inc_angle'].dropna(inplace=False).mean()
    df['inc_angle'] = df['inc_angle'].fillna(mean_inc_angle)
    df['band_1'] = df['band_1'].astype(list)
    df['band_2'] = df['band_2'].astype(list)
    return df
In [*]:

test_df = pd.read_json(TEST_PATH)
In [*]:

test_df = clean_dataframe(test_df)
In [43]:

def draw_sample(sample, ncols=4, titles=None):
    nrows = np.ceil(1. * len(sample) / ncols)    
    plt.figure(figsize=(16, 4 * nrows))
    for j, img_raw in enumerate(sample):
        img = np.reshape(img_raw, (75,75))
        plt.subplot(nrows, ncols, j+1)
        plt.imshow(img, cmap='jet')
        if titles: plt.title(titles[j])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
In [ ]:

sample_icebergs = train_df[train_df['is_iceberg'] == 1].sample(4)
sample_ships = train_df[train_df['is_iceberg'] == 0].sample(4)
print("Iceberg::Band-1")
draw_sample(sample_icebergs['band_1'], titles=sample_icebergs['id'].tolist())
print("Iceberg::Band-2")
draw_sample(sample_icebergs['band_2'], titles=sample_icebergs['id'].tolist())
print("Ship::Band-1")
draw_sample(sample_ships['band_1'], titles=sample_ships['id'].tolist())
print("Ship::Band-2")
draw_sample(sample_ships['band_2'], titles=sample_ships['id'].tolist())
