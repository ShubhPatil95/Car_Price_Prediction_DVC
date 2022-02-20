import argparse
from email.policy import default
import yaml
import pandas as pd
import os


def read_file(file_path):
    with open(file_path) as yaml_file:
        file=yaml.safe_load(yaml_file)
    return file

def get_data(paramsyaml_path,pathyaml_path):
    config_params=read_file(paramsyaml_path)
    config_paths=read_file(pathyaml_path)
    
    source_data_path=config_paths["data"]["source_data"]
    raw_data=config_paths["data"]["raw_data"]

    df=pd.read_csv(source_data_path)
    df_col_name=[col.replace(" ","_") for col in df.columns]

    df.to_csv(raw_data,columns=df_col_name)
  

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config_path",default="paths.yaml")
    args.add_argument("--config_params",default="params.yaml")
    parsed_args = args.parse_args()
    get_data(paramsyaml_path=parsed_args.config_params,pathyaml_path=parsed_args.config_path)
