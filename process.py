import json
import os
import pickle as pk

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path


import logging

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import models

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input filename", required=True)
    parser.add_argument("-o", "--output", help="output filename", required=True)
    args = parser.parse_args()
    data = json.load(open(args.input, "r"))
    id2label = models.cluster_classification(data, train_portion=0.4)
    df = pd.DataFrame.from_dict(data, orient='index')
    df['label'] = df.index.to_series().apply(lambda x: id2label.get(x))
    df.to_json(args.output, orient='index', force_ascii=False, indent=2)
