import os
import sys
import re
import opensmile
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from src.config import *
import pandas as pd

project_root = os.path.dirname('..')
sys.path.append(project_root)



full_ucl_data = glob.glob(f'{UCL_DATA}/*.wav')
files = [f for f in full_ucl_data  if 'channel1' in f] 

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def process(f):
    return smile.process_file(f)

# Run in parallel with progress bar
results = Parallel(n_jobs=18)(
    delayed(process)(f) for f in tqdm(files, desc="Extracting features")
)

features = pd.concat(results)
features['trials'] = [re.search(r'trial(\d+)', f).group(1) for f in features.index.get_level_values('file')]
features['speaker'] = [re.search(r'speaker(\d+)', f).group(1) for f in features.index.get_level_values('file')]
features["sample_id"] = [
    next((k for k, v in MAPPING.items() if v == f"speaker{speaker}"), None)
    for speaker in features["speaker"].values
]
features = features.set_index(['trials', 'speaker', 'sample_id'])
features.to_csv(f"{OUTPUT_FOLDER}/opensmile_features_compare2016.csv")