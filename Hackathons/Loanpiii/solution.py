import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


trainingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/Loanpiii/train.csv"
testingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/Loanpiii/test.csv"

encode_features = ['Gender','Married','Education','Self_Employed','Dependents','Loan_Status']
fillna_withmean = ['LoanAmount','Loan_Amount_Term']
fillna_withmostcommon = ['Dependents','Gender','Credit_History','Married','Self_Employed']


def transformData(data):
	df = data

	for feature in fillna_withmean:
		if feature in df.columns.values:
			df[feature] = df[feature].fillna(df[feature].mean())

	for feature in fillna_withmostcommon:
		if feature in df.columns.values:
			df[feature] = df[feature].fillna(df[feature].value_counts().index[0])

	for feature in encode_features:
		if feature in data.columns.values:
			le = LabelEncoder()
			df[feature] = le.fit_transform(df[feature])

	df['Household_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
	df = df.drop(['ApplicantIncome','CoapplicantIncome'],axis=1)
	dummies = pd.get_dummies(df.Property_Area)
	df = pd.concat([df,dummies],axis=1)
	df = df.drop('Property_Area',axis=1)

	return df



def functionLoanPrediction():
	train_df = pd.read_csv(trainingFile,index_col=0)
	test_df = pd.read_csv(testingFile,index_col=0)
	train_df = transformData(train_df)
	test_df = transformData(test_df)
	col_names = train_df.columns.tolist()

	train_df.insert(len(train_df.columns)-1,'Loan_Status',train_df.pop('Loan_Status'))
	scale_features = ['LoanAmount','Loan_Amount_Term','Household_Income']

	train_df[scale_features] = train_df[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)
	test_df[scale_features] = test_df[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)
	#train_df = (train_df-train_df.mean())/(train_df.max()-train_df.min())
	#test_df = (test_df-test_df.mean())/(test_df.max()-test_df.min())


	xTrain = train_df.iloc[:, :-1]
	yTrain = train_df.iloc[:,-1]
	xTest = test_df



	lr = LogisticRegression()
	lr = lr.fit(xTrain, yTrain)
	yTest = lr.predict(xTest)

	yTest = ['Y' if x==1 else 'N' for x in yTest]
	xTest['Loan_Status']=yTest
	xTest = xTest['Loan_Status']
	xTest.to_csv('/home/ashish/Documents/AnalyticsVidhya/Hackathons/Loanpiii/loans_submission.csv',sep=',',header=True)





if __name__ == '__main__':
	functionLoanPrediction()