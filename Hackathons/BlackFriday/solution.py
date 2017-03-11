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


trainingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/BlackFriday/train.csv"
testingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/BlackFriday/test.csv"


def transformData(data):
	pass


def applyModel():
	train_df = pd.read_csv(trainingFile)
	test_df = pd.read_csv(testingFile)
	


if __name__ == '__main__':
	applyModel()