import pandas as pd 
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

trainingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/Loanpiii/train.csv"
testingFile = "/home/ashish/Documents/AnalyticsVidhya/Hackathons/Loanpiii/test.csv"

def cleaningtrainDF():
	trainDF = pd.read_csv(trainingFile, index_col="Loan_ID")
	print trainDF.shape
	sampleDF = trainDF.loc[(trainDF["Gender"]=="Female") & (trainDF["Education"]=="Not Graduate") & (trainDF["Loan_Status"]=="Y"), ["Gender", "Education", "Loan_Status"]]
	#print sampleDF.head()
	print trainDF.dtypes

	colTypes = pd.read_csv("datatype.csv")
	print colTypes


	# for i , row in colTypes.iterrows():
	# 	if row["type"]=="categorical":
	# 		trainDF[row["feature"]] = trainDF[row["feature"]].astype(np.object)
	# 	elif row["type"]=="continuous":
	# 		trainDF[row["feature"]] = trainDF[row["feature"]].astype(np.float)


	for i, row in colTypes.iterrows():  #i: trainDFframe index; row: each row in series format
		if row['type']=="categorical":
			trainDF[row['feature']]=trainDF[row['feature']].astype(np.object)
		elif row['type']=="continuous":
			trainDF[row['feature']]=trainDF[row['feature']].astype(np.float)

	print trainDF.dtypes

	#print trainDF.apply(lambda x: sum(x.isnull()), axis=1).head() 

	print mode(trainDF["Gender"])[0]

	trainDF['Gender'].fillna(mode(trainDF['Gender'])[0], inplace=True)
	print trainDF.apply(lambda x: sum(x.isnull()), axis=0)
	print trainDF["Gender"].value_counts()

	#impute_grps = pd.pivot_table(trainDF, values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
	#print impute_grps

	print trainDF.apply(lambda x: sum(x.isnull()), axis=0)

	#tempDF =  trainDF.sort_values(['ApplicantIncome','CoapplicantIncome'],ascending=False)
	#print tempDF.head()

	#trainDF.boxplot(column="ApplicantIncome",by="Loan_Status")
	#trainDF.hist(column="ApplicantIncome",by="Loan_Status",bins=30)

	#a = trainDF["Gender"].tolist()
	#print a[:5]

	# b = pd.Series(trainDF["Gender"], copy=True)
	# print type(b)
	# print b

	trainDF["Loan_Status_Coded"] = coding(trainDF["Loan_Status"], {"N":0,"Y":1})

	print trainDF["Loan_Status_Coded"].value_counts()
	print trainDF["Loan_Status"].value_counts()

def coding(col, dict):

	columnCoded = pd.Series(col, copy=True)
	for key, value in dict.items():
		columnCoded.replace(key, value, inplace=True)
	return columnCoded

def dataFrame():
	data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
	print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))


	# Take a 2D array as input to your DataFrame 
	my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
	print(pd.DataFrame(my_2darray))

	# Take a dictionary as input to your DataFrame 
	my_dict = {1: ['1', '3'], 2: ['1', '2'], 3: ['2', '4']}
	print(pd.DataFrame(my_dict))

	# Take a DataFrame as input to your DataFrame 
	my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['A'])
	print(pd.DataFrame(my_df))

	# Take a Series as input to your DataFrame
	my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington", "Belgium":"Brussels"})
	print(pd.DataFrame(my_series))

	df.reset_index(level=0, drop=True)

	# Append a column to `df`
	df.loc[:, 4] = pd.Series(['5', '6'], index=df.index)

	temp = df.replace(['Awful', 'Poor', 'OK', 'Acceptable', 'Perfect'], [0, 1, 2, 3, 4])

#pd.__version__  pd.show_versions(as_json=False)
# ipython nbconvert --to python Untitled0.ipynb

if __name__ == '__main__':
	cleaningtrainDF()

