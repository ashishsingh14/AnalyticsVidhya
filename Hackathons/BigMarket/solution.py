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


trainingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/BigMarket/train.csv"
testingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/BigMarket/test.csv"

def cleaningData():
	train = pd.read_csv(trainingFile)
	test = pd.read_csv(testingFile)
	train['source'] = "train"
	test['source'] = "test"
	data = pd.concat([train,test],ignore_index=True)
	print data.shape, train.shape,test.shape
	


if __name__ == '__main__':
	cleaningData()