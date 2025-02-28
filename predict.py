import torch
import torch.nn as nn
import pandas as pd
import os
from train import MLP
from sklearn.preprocessing import StandardScaler

def predict_on_submit_data():
    model.eval()
    with torch.no_grad():
        y_submit_test= model(X_submit)
    return scaler_y.inverse_transform(y_submit_test.cpu().numpy())

write_path = 'solution_template/pred.csv'
prediction_df = pd.read_csv(write_path)
first=True
dropColumns= ["date","altitude","year"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for fichier in os.listdir('interpolatedData'):
    if ".ipynb_checkpoints" not in fichier:
        nom_region = fichier[:-4]
        interpolatedReg = pd.read_csv('interpolatedData/'+fichier,index_col=0)
        interpolatedReg['date'] = pd.to_datetime(interpolatedReg['date'])
        interpolatedReg = interpolatedReg.sort_values("date")
        
        split_date ='2021-12-31 22:30:00'
        known_set = interpolatedReg.loc[interpolatedReg['date'] <= split_date].drop(columns=dropColumns)
        known_set.dropna(axis=0,how='any',inplace=True)
        unknown_set =interpolatedReg.loc[interpolatedReg['date'] > split_date].drop(columns=dropColumns)

        
        data = known_set.values
        X = data[:, :-1]  # Features
        y = data[:, -1].reshape(-1, 1)
            
        X_submit = unknown_set.values[:, :-1]
            
            # Standardize the dataset
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
            
    
        X_submit = scaler_X.transform(X_submit)
        X_submit = torch.tensor(X_submit).float()
        X_submit = X_submit.to(device)
        
        model = torch.load('./models/'+nom_region, weights_only=False)
    
        predicted_submit_loads = predict_on_submit_data()
            
        submit_loads=unknown_set['estimated_load'].to_numpy()+predicted_submit_loads[:,0]
            
        
        write_column = 'pred_'+nom_region
            
        prediction_df[write_column] = submit_loads
        if "Métropole" not in fichier:
            if first :
                prediction_df["pred_France"]=prediction_df[write_column]
                first=False
            else:
                prediction_df["pred_France"]=prediction_df["pred_France"]+prediction_df[write_column]
        
prediction_df.to_csv(write_path,index=False,index_label=False)