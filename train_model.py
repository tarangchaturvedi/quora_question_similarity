import os
import pandas as pd
import warnings
from simple_features import simple_featurizing, extract_features
from data_preprocessing import preprocess_data
from utility import create_word2vec_features
from model_training import final_model

def train_model(file_path):
    # suppress warnings
    warnings.simplefilter(action='ignore')

    print("training")

    if os.path.isfile("/mnt/e/projects/Quora question_pair similarity/simple_features.csv"):
        df_sf = pd.read_csv("/mnt/e/projects/Quora question_pair similarity/simple_features.csv")
    else:
        df_sf = simple_featurizing(file_path)
        df_sf.to_csv("/mnt/e/projects/Quora question_pair similarity/simple_features.csv")

    if os.path.isfile("/mnt/e/projects/Quora question_pair similarity/preprocess_data.csv"):
        df_pd = pd.read_csv("/mnt/e/projects/Quora question_pair similarity/preprocess_data.csv")
    else:
        df_pd = preprocess_data(file_path)
        df_pd.to_csv("/mnt/e/projects/Quora question_pair similarity/preprocess_data.csv")
    
    if os.path.isfile("/mnt/e/projects/Quora question_pair similarity/nlp_features.csv"):
        df_nlp = pd.read_csv("/mnt/e/projects/Quora question_pair similarity/nlp_features.csv")
    else:
        df_nlp = extract_features(df_pd)
        df_nlp.to_csv("/mnt/e/projects/Quora question_pair similarity/nlp_features.csv")

    if os.path.isfile("/mnt/e/projects/Quora question_pair similarity/w2v_features.csv"):
        df_w2v = pd.read_csv("/mnt/e/projects/Quora question_pair similarity/w2v_features.csv")
    else:
        df_w2v = create_word2vec_features(df_pd)
        df_w2v.to_csv("/mnt/e/projects/Quora question_pair similarity/w2v_features.csv")

    if os.path.isfile("/mnt/e/projects/Quora question_pair similarity/final_features.csv"):
        df_ff = pd.read_csv("/mnt/e/projects/Quora question_pair similarity/final_features.csv")
    else:
        df_ff = final_features(df_nlp, df_sf, df_w2v)
        df_ff.to_csv("/mnt/e/projects/Quora question_pair similarity/final_features.csv")

    final_model(df_ff)