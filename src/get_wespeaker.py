import os
import sys
import re
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import wespeaker

project_root = os.path.dirname('..')
sys.path.append(project_root)

from src.config import *

# Collect files
full_ucl_data = glob.glob(f'{UCL_DATA}/*.wav')
files = [f for f in full_ucl_data if 'channel1' in f and 'trial10' in f] # trial 10 is the trial from UCL-SBM dataset used for the Compare2020 interspeech challenge

# Load model once
GLOBAL_WESPEAKER_MODEL = wespeaker.load_model('english')

def process(f):
    emb = GLOBAL_WESPEAKER_MODEL.extract_embedding(f)
    emb = emb.detach().cpu().numpy().squeeze() 

    trial = re.search(r'trial(\d+)', f).group(1)
    speaker = re.search(r'speaker(\d+)', f).group(1)
    sample_id = next((k for k, v in MAPPING.items() if v == f"speaker{speaker}"), None)
    emb_dict = {f"wespeaker_{i}": v for i, v in enumerate(emb)}

    return {
        "trials": trial,
        "speaker": speaker,
        "sample_id": sample_id,
        **emb_dict
    }

results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(process)(f) for f in tqdm(files, desc="Extracting WeSpeaker features")
)

features = pd.DataFrame(results)
features = features.set_index(['trials', 'speaker', 'sample_id'])
features.to_csv(f"{OUTPUT_FOLDER}/wespeaker_features.csv")
