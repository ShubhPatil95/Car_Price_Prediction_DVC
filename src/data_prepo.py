import pandas as pd
import argparse
from get_data import read_file
import warnings
warnings.filterwarnings("ignore")

def data_prepo(pathyaml_path):
    config_path=read_file(pathyaml_path)
    raw_data=config_path["data"]["raw_data"]
    processed_data=config_path["data"]["processed_data"]
    
    df=pd.read_csv(raw_data)

    final_dataset=df[["Year","Selling_Price","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]]
    final_dataset["2020"]=2020
    final_dataset["no_year"]=final_dataset["2020"]-final_dataset["Year"]
    final_dataset=final_dataset.drop(["Year","2020"],axis=1)

    ## One Hot Encoding
    final_dataset=pd.get_dummies(final_dataset,drop_first=True) 

    ##export processed data
    final_dataset.to_csv(processed_data)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config_paths",default="paths.yaml")
    args_parse=args.parse_args()
    data_prepo(args_parse.config_paths)



    
