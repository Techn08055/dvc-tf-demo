import yaml
import os
import time
import json

def read_yaml(path_to_yaml:str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content

def get_time_stamp (name):
    time_stamp = time.asctime().replace(" ", "_").replace(":","_")
    uniquename  = f"{name}_at_{time_stamp}"
    return uniquename

def create_directory(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok = True)
        print((dir_path))

def save_local_df(data, data_path,index_status= False):
    data.to_csv(data_path, index = index_status)
    print(f"data is savedin {data_path}")

def save_reports(scors:dict, scores_path:str):
    with open(scores_path, "w") as f:
        json.dump(scors, f, indent =4)
    print(f"reports are saved at {scores_path}")