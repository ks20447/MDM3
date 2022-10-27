#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:13:11 2022

@author: conradodriscoll
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




def Pre_Process_Data_W_Target(file_path):
    
    data = pd.read_csv(file_path)
    df = data.copy()
    df_relevant = df[['Client_Income','House_Own','Credit_Amount','Loan_Annuity','Client_Marital_Status','Client_Gender','Age_Days','Employed_Days','Client_Family_Members', 'Default']]
    RemovedNull = df_relevant.dropna()
    RemovedNull1 = RemovedNull.astype({'Credit_Amount':'float'})
    RemovedNull1 = RemovedNull1.astype({'Loan_Annuity':'float'})
    RemovedNull1 = RemovedNull1.astype({'Client_Income':'float'})
    RemovedNull1 = RemovedNull1.astype({'House_Own':'float'})
    RemovedNull1 = RemovedNull1.astype({'Age_Days':'float'})
    RemovedNull1 = RemovedNull1.astype({'Employed_Days':'float'})
    RemovedNull1 = RemovedNull1.astype({'Client_Family_Members':'float'})
    RemovedNull1 = RemovedNull1.astype({'Default':'float'})
        
    
    train_data = RemovedNull1.copy()
    
    marital_dict = {"M":0 , "W":1 , "S": 2 , "D":3}

    gender_dict = {"Male":0, "Female":1}


    train_data['Client_Marital_Status'] = (train_data['Client_Marital_Status'].map(marital_dict))

    train_data['Client_Gender'] = (train_data['Client_Gender'].map(gender_dict))
    
    
    
    X = train_data.iloc[:,[0,1,2,3,4,5,6,7,8]]
    y = train_data.iloc[:,9]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    
    

    return X_train, X_test, y_train, y_test


def Pre_Process_Data_WO_Target(file_path):
    
    data = pd.read_csv(file_path)
    df = data.copy()
    df_relevant = df[['Client_Income','House_Own','Credit_Amount','Loan_Annuity','Client_Marital_Status','Client_Gender','Age_Days','Employed_Days','Client_Family_Members']]
    RemovedNull = df_relevant.dropna()
    RemovedNull1 = RemovedNull.astype({'Credit_Amount':'float'})
    RemovedNull1 = RemovedNull1.astype({'Loan_Annuity':'float'})
    RemovedNull1 = RemovedNull1.astype({'Client_Income':'float'})
    RemovedNull1 = RemovedNull1.astype({'House_Own':'float'})
    RemovedNull1 = RemovedNull1.astype({'Age_Days':'float'})
    RemovedNull1 = RemovedNull1.astype({'Employed_Days':'float'})
    RemovedNull1 = RemovedNull1.astype({'Client_Family_Members':'float'})        
    
    testing_data = RemovedNull1.copy()
    
    marital_dict = {"M":0 , "W":1 , "S": 2 , "D":3}

    gender_dict = {"Male":0, "Female":1}


    testing_data['Client_Marital_Status'] = (testing_data['Client_Marital_Status'].map(marital_dict))

    testing_data['Client_Gender'] = (testing_data['Client_Gender'].map(gender_dict))
    
    
    
    X_unknown_pred = testing_data.iloc[:,0:9]
    
    
    
    
    return X_unknown_pred


    
#X_train1, X_test1, y_train1, y_test1 = Pre_Process_Data("/Users/conradodriscoll/Desktop/University/3rd_Year/MDM3/Sopra/Datasets/archive/Train_Dataset.csv")


def Create_Model(Train_Data_Path):
    
    X_train, X_test, y_train, y_test = Pre_Process_Data_W_Target(Train_Data_Path)
    
    classifier = LogisticRegression()

    classifier.fit(X_train,y_train)

    return classifier




def Test_Classifier_Accuracy(Train_Data_Path):
    classifier = Create_Model(Train_Data_Path)
    
    X_train, X_test, y_train, y_test = Pre_Process_Data_W_Target(Train_Data_Path)
    
    y_pred = classifier.predict(X_test)
    
    Conf_Matrix = confusion_matrix(y_test, y_pred)
    print(Conf_Matrix)

    Accuracy_Score = accuracy_score(y_test, y_pred)
    print(Accuracy_Score)
    

def Test_Classifier_Bands(Train_Data_Path):
    classifier = Create_Model(Train_Data_Path)
    X_train, X_test, y_train, y_test = Pre_Process_Data_W_Target(Train_Data_Path)
    
    
    y_pred = classifier.predict(X_test)
    
    
    Probability = classifier.predict_proba(X_test)
    
    df_prediction_prob = pd.DataFrame(Probability, columns = ['prob_0', 'prob_1'])
    df_prediction_target = pd.DataFrame(y_pred, columns = ['predicted_TARGET'])
    df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

    dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
    
    df1 = dfx.sort_values(
            by="prob_0",
            ascending=False)
    
    decile_pre_divide= len(dfx) - len(dfx) % 40
    
    decile_pre_divide2 = decile_pre_divide / 40
    
    splitter = decile_pre_divide2
    

    Band12 = df1.iloc[[1*splitter],[1]].values
    Band13 = df1.iloc[[2*splitter],[1]].values
    Band14 = df1.iloc[[3*splitter],[1]].values
    Band15 = df1.iloc[[4*splitter],[1]].values
    Band16 = df1.iloc[[5*splitter],[1]].values
    Band17 = df1.iloc[[6*splitter],[1]].values
    Band18 = df1.iloc[[7*splitter],[1]].values
    Band19 = df1.iloc[[8*splitter],[1]].values
    Band20 = df1.iloc[[9*splitter],[1]].values

    Bands = [Band12,Band13,Band14,Band15,Band16,Band17,Band18,Band19,Band20]
    
    print(Bands)
    return Bands
    



#Xt = Initialise_Model_From_Train_Data("/Users/conradodriscoll/Desktop/University/3rd_Year/MDM3/Sopra/Datasets/archive/Train_Dataset.csv")

#taking in rando plurs

def Predict_From_Generated_Customer(Predict_Data_Path, Train_Data_Path):
    classifier = Create_Model(Train_Data_Path)
    
    Bands = Test_Classifier_Bands(Train_Data_Path)
    X_unknown_pred = Pre_Process_Data_WO_Target(Predict_Data_Path)
    df = classifier.predict(X_unknown_pred)
    df_proba = classifier.predict_proba(X_unknown_pred)
    
    
    
    
    
    
    X_unknown_pred['prob_0'] = df_proba[:,[0]]
    X_unknown_pred['Band'] = 10
    
    
    iteration_length = len(X_unknown_pred)
    
    for i in range(iteration_length):
        
        locator = int(i)
        
        predicted_probability = X_unknown_pred.iloc[locator,9]
        
        band_to_assign = 10      
        if float(predicted_probability) >= float(Bands[0]):
            band_to_assign = 1
        
        elif float(predicted_probability) >= float(Bands[1]):
            band_to_assign = 2
            
        elif float(predicted_probability) >= float(Bands[2]):
            band_to_assign = 3
        
        elif float(predicted_probability) >= float(Bands[3]):
            band_to_assign = 4
        
        elif float(predicted_probability) >= float(Bands[4]):
            band_to_assign = 5
        
        elif float(predicted_probability) >= float(Bands[5]):
            band_to_assign = 6
        
        elif float(predicted_probability) >= float(Bands[6]):
            band_to_assign = 7
        
        elif float(predicted_probability) >= float(Bands[7]):
            band_to_assign = 8
        
        elif float(predicted_probability) >= float(Bands[8]):
            band_to_assign = 9
        
        else:
            continue
        
        X_unknown_pred.iloc[[locator],[10]] = band_to_assign

    return X_unknown_pred
        
        
            
                
        
       

        
            
            
    
    
    
xd = Predict_From_Generated_Customer(#GENERATED CUSTOMERS CSV ,#TRAIN SET )

                                     
    
    
x = xd['Band'].value_counts()


    
    
    
    
    
    
    
    