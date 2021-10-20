import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression 

path_train = r"Dataset\train.csv"
train = pd.read_csv(path_train)

train['Gender'] = train['Gender'].fillna(train['Gender'].dropna().mode().values[0])
train['Married'] = train['Married'].fillna(train['Married'].dropna().mode().values[0])
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].dropna().mode().values[0])
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].dropna().mode().values[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].dropna().mode().values[0])
train['Self_Employed'] = train['Self_Employed'].fillna(train['Self_Employed'].dropna().mode().values[0])
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].dropna().mode().values[0])


token = {"Male": 1, "Female": 2,
         "Yes": 1, "No" : 2, 
         "Graduate": 1, "Not Graduate": 2, 
         "Urban": 3, "Semiurban": 2, "Rural": 1,
         "3+": 3, "2": 2, "1":1, "0":0,
         "Y": 1, "N": 0}


train = train.applymap(lambda s: token.get(s) if s in token else s)

train.drop("Loan_ID", axis=1 , inplace = True)

y = train["Loan_Status"]
X = train.drop("Loan_Status",axis=1)

log_reg = LogisticRegression()

log_reg.fit(X,y)

ID,Gender, Married, Dep, Edu, Employed, App_Inc, Co_app_inc, Amt, Term, Cred_hist, Area= [],[],[],[],[],[],[],[],[],[],[],[]
for _ in range(int(input("Please Enter No. Of Enteries : "))):
    id = input("Please enter Applicant Loan ID: ")
    ID.append(id)
    gen = input("Please enter Gender : Male           Female : ")
    Gender.append(gen)
    married = input("Marital Status  : Y for Married           N for Unmarried : ")
    Married.append(married)
    dep = input("Please enter No. of Dependents (Note :- In case of 3 or More Dependents Please Enter '3+') : ")
    Dep.append(dep)
    edu = input("Please enter Education : Graduate or Not Graduate : ")
    Edu.append(edu)
    employ = input("Are you Self Employed : Y for Yes           N for No : ")
    Employed.append(employ)
    Appinc = int(input("Please enter Applicant Income: "))
    App_Inc.append(Appinc)
    coAppinc = int(input("Please enter CoApplicant Income: "))
    Co_app_inc.append(coAppinc)
    amt = int(input("Please Enter Loan Amount : "))
    Amt.append(amt)
    term = int(input("Please Enter Term Of Loan Amount Repayment : "))
    Term.append(term)
    Credhist = int(input("Please Enter 0 for Bad and 1 for Good Credit History : "))
    Cred_hist.append(Credhist)
    area = input("Please enter Property Type : Urban        Semiurban           Rural : ")
    Area.append(area)
    
df = pd.DataFrame({'Gender':Gender,
                   'Married': Married,
                   'Dependents':Dep,
                   'Education': Edu,
                   'Self_Employed': Employed,
                   'ApplicantIncome': App_Inc,
                   'CoapplicantIncome': Co_app_inc,
                   'LoanAmount': Amt,
                   'Loan_Amount_Term': Term,
                   'Credit_History': Cred_hist,
                   'Property_Area':Area
})

df.to_csv("Pred.csv",index = False)

path = r"Pred.csv"
Pred_data = pd.read_csv(path)

pred_data_origonal = Pred_data.copy()

pred_data = Pred_data.applymap(lambda s: token.get(s) if s in token else s)

pred = log_reg.predict(pred_data)

for i in range(len(pred)):
    p = lambda x: "Y" if (x==1) else "N"
    print("Loan ID : ",ID[i])
    print("Loan Approved : ", p(pred[i]))
