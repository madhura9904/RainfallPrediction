import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

from src.components.data_transformation import DataTransformation

data_transformation=DataTransformation()

train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
