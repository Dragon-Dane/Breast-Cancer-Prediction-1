import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
from scipy.stats.stats import pearsonr
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO  
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#from mlxtend.plotting import plot_decision_regions



#Loading Data
os.chdir("C:\\Users\\hieut\\OneDrive\\OSU\\5223-Programming for Data Science\\Project")
cancer_data = pd.read_csv('project.csv')

cancer_data.columns.values
cancer_data.isnull().sum()
cancer_data['diagnosis'] = cancer_data['diagnosis'].map({'M':1,'B':0})
cancer_data.head()
cancer_data.drop('id',axis=1,inplace=True)
len(cancer_data.columns)
cancer_data.loc[:,'radius_mean': 'fractal_dimension_worst']
diagnosis = cancer_data.loc[:,'diagnosis']
diagnosis


#Standardizing Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cancer_data.loc[:,'radius_mean': 'fractal_dimension_worst'])
scaled_data = pd.DataFrame(cancer_data, columns=[ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_se', 'texture_se',
       'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
       'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'])
scaled_data

data = pd.concat([diagnosis, scaled_data], axis=1)
data


#Count of B and M
data['diagnosis'].value_counts()
sns.countplot( x= 'diagnosis', data = data, palette = 'hls')
plt.show()
plt.savefig('diagnosis_count_plot')


#Percentage study of B and M
count_B = len(data[data['diagnosis']=='B'])
count_M = len(data[data['diagnosis']=='M'])
pct_of_B = count_B/(count_B+count_M)
print("percentage of B", pct_of_B*100)
pct_of_M = count_M/(count_B+count_M)
print("percentage of M", pct_of_M*100)




#Checking multicollenarity with VIF
P = add_constant(data)
vif = pd.Series([variance_inflation_factor(P.values, i) 
               for i in range(P.shape[1])], 
              index=X.columns)
print(vif)


#Comparing man, median, std

a = data.iloc[:,1:11]
b = data.iloc[:,11:21]
c= data.iloc[:,21:31]



#Splitting Data
data
X = data.iloc[:,1:11]
y = data.iloc[:,0]
N = 569
traindf, testdf = train_test_split(data, test_size = 0.25)
var_traindf = traindf.var(ddof=1)
var_tesrdf = traindf.var(ddof=1)
std_dev = np.sqrt(( var_traindf + var_tesrdf )/2)
std_dev
t = (traindf.mean() - testdf.mean())/(std_dev*np.sqrt(2/N))
df = 2*N - 2
p = 1 - sts.t.cdf(t,df=df)
print("t = " + str(t))
print("p = " + str(2*p))
# Cross Checking with the internal scipy function
t2, p2 = sts.ttest_ind(traindf,testdf)
print("t = " + str(t2))
print("p = " + str(p2))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.25)



#Decision Tree Gini
gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 
gini.fit(X_train, y_train) 

gini_data = StringIO()
export_graphviz(gini, out_file= gini_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(gini_data.getvalue())  
Image(graph.create_png())

y_pred_gini = gini.predict(X_test) 
print("Predicted values:") 
print(y_pred_gini) 
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_gini)) 
print ("Accuracy : ", accuracy_score(y_test,y_pred_gini)*100) 
print("Report : ", classification_report(y_test, y_pred_gini)) 

#False postive rate, True positive rate, calculating AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, gini.predict_proba(X_test)[:,1])
auc = metrics.roc_auc_score(y_test,gini.predict(X_test))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Decision Tree Gini", auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 


#Decision Tree Entropy
entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5) 
entropy.fit(X_train, y_train) 

entropy_data = StringIO()
export_graphviz(entropy, out_file=entropy_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(entropy_data.getvalue())  
Image(graph.create_png())


y_pred_entropy = entropy.predict(X_test) 
print("Predicted values:") 
print(y_pred_entropy) 
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_entropy)) 
print ("Accuracy : ", accuracy_score(y_test,y_pred_entropy)*100) 
print("Report : ", classification_report(y_test, y_pred_entropy)) 

#False postive rate, True positive rate, calculating AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, entropy.predict_proba(X_test)[:,1])
auc = metrics.roc_auc_score(y_test,entropy.predict(X_test))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Decision Tree Entropy", auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 



#Logistic regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())
y_pred_log_reg = log_reg.predict(X_test) 
print("Predicted values:") 
print(y_pred_log_reg) 
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_log_reg)) 
print ("Accuracy : ", accuracy_score(y_test,y_pred_log_reg)*100) 
print("Report : ", classification_report(y_test, y_pred_log_reg)) 

#False postive rate, True positive rate, calculating AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, log_reg.predict_proba(X_test)[:,1])
auc = metrics.roc_auc_score(y_test,log_reg.predict(X_test))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Logistic Regression", auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 



#Support Vector Machines

svm_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
Support_Vector_Machine = CalibratedClassifierCV(svm_model) 
Support_Vector_Machine.fit(X_train, y_train)
print(Support_Vector_Machine.score(X_train, y_train))
y_pred_svm = Support_Vector_Machine.predict(X_test)
print("Predicted values:") 
print(y_pred_svm) 
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_svm)) 
print ("Accuracy : ", accuracy_score(y_test,y_pred_svm)*100) 
print("Report : ", classification_report(y_test, y_pred_svm)) 

#False postive rate, True positive rate, calculating AUC
fpr, tpr, thresholds = metrics.roc_curve(y_test, Support_Vector_Machine.predict_proba(X_test)[:,1])
auc = metrics.roc_auc_score(y_test,Support_Vector_Machine.predict(X_test))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("Support Vector Machine", auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 
