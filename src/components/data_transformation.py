import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from src.utils import save_object
from src.exception import CustomException
from src.logger import logger
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Data Transformation

        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                'gender', 
                'race_ethnicity',
                'parental_level_of_education',
                'lunch', 
                'test_preparation_course'
                ]
            num_pipeline=Pipeline(
               steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
               ] 
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=True)),  # or sparse=True in older versions
                    ("scaler", StandardScaler(with_mean=False))
                ] 
            )

            logger.info("Categorical Columns Encoding Completed")
            logger.info("Numericals Columns Standard Scaling Completed")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path)  :
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logger.info("Read Train and test data completed")
            logger.info("Obtating Preprocessing  object")
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_columns = ['writing_score', 'reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logger.info(
                f"Applying Preprocessing object on training dataframe and testing dataframe"
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logger.info(
                f"Saved Processing object"
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
       
        except Exception as e:
            raise CustomException(e, sys)
