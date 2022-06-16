#!/usr/bin/env python
# coding: utf-8

# # Homework 3
# 
# ## Part 1: Imbalanced Dataset
# This part of homework helps you practice to classify a highly imbalanced dataset in which the number of examples in one class greatly outnumbers the examples in another. You will work with the Credit Card Fraud Detection dataset hosted on Kaggle. The aim is to detect a mere 492 fraudulent transactions from 284,807 transactions in total. 
# 
# ### Instructions
# 
# Please push the .ipynb, .py, and .pdf to Github Classroom prior to the deadline. Please include your UNI as well.
# 
# Due Date : TBD
# 
# ### Name: Rakshith Kamath
# 
# ### UNI: rk3165
# 
# ## 0 Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_validate
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve,auc,average_precision_score


# ## 1 Data processing and exploration
# Download the Kaggle Credit Card Fraud data set. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# In[2]:


raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()


# ### 1.1 Examine the class label imbalance
# Let's look at the dataset imbalance:
# 
# **Q1. How many observations are there in this dataset? How many of them have positive label (labeled as 1)?**

# In[3]:


# Your Code Here
print(f'The number of observations in this dataset is: {raw_df.shape[0]}')
class_count=raw_df.groupby(['Class'])['Class'].count()
print(f'Out of which, count for class 0 is :{class_count[0]}')
print(f'and count for class 1 is: {class_count[1]}')


# ### 1.2 Clean, split and normalize the data
# The raw data has a few issues. First the `Time` and `Amount` columns are too variable to use directly. Drop the `Time` column (since it's not clear what it means) and take the log of the `Amount` column to reduce its range.

# In[4]:


cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)


# **Q2. Split the dataset into development and test sets. Please set test size as 0.2 and random state as 42.**

# In[5]:


# Your Code Here
class_y = cleaned_df['Class']
cleaned_df.pop('Class')
cleaned_df


# In[6]:


X_dev, X_test, y_dev, y_test = train_test_split(cleaned_df, class_y,
                                                stratify=class_y, 
                                                test_size=0.2,
                                                random_state=42)
features = raw_df.columns


# **Q3. Normalize the input features using the sklearn StandardScaler. Print the shape of your development features and test features.**

# In[7]:


# Your Code Here
preprocess= StandardScaler()
X_dev = preprocess.fit_transform(X_dev,y_dev)
X_test = preprocess.transform(X_test)
print(f'Shape of the development features matrix {X_dev.shape}')
print(f'Shape of the test features matrix {X_test.shape}')


# ### 1.3 Define the model and metrics
# **Q4. First, fit a default logistic regression model. Print the AUC and average precision of 5-fold cross validation.**

# In[8]:


# Your Code Here
# Fitting the logistic model
scores_logReg= cross_validate(LogisticRegression(),
                       X_dev,y_dev,cv=5,
                       scoring=['roc_auc','average_precision'],
                       return_estimator= True)
print(f"AUC score-{scores_logReg['test_roc_auc']}")
print(f"Average Precision score-{scores_logReg['test_average_precision']}")
print(f"Mean AUC score-{scores_logReg['test_roc_auc'].mean()}")
print(f"Mean Average Precision score-{scores_logReg['test_average_precision'].mean()}")


# In this case there are 5 models with the scores of each for various metrics as shown above. We select the model for further study based on the model that gave us the maximum AUC score. Given this assumption, we calculate and check to get the index of the best model from the given set of 5 models.We have done this after consultation with TA.

# In[9]:


log_best_index=np.argmax(scores_logReg['test_roc_auc'])


# **Q5.1. Perform random under sampling on the development set. What is the shape of your development features? How many  positive and negative labels are there in your development set? (Please set random state as 42 when performing random under sampling)**

# In[10]:


# Your Code Here
rus = RandomUnderSampler(random_state=42,replacement=False)
X_under, y_under = rus.fit_resample(X_dev, y_dev)
print(f'Shape of the development features matrix {X_under.shape}')
y_under= np.array(y_under)
non_zero_count = np.count_nonzero(y_under)
print(f'Out of which, count for class 0 is :{len(y_under)-non_zero_count}')
print(f'and count for class 1 is: {non_zero_count}')


# **Q5.2. Fit a default logistic regression model using under sampling. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing random under sampling)**

# In[11]:


pipe = imb_make_pipeline(rus,LogisticRegression())
scores_under = cross_validate(pipe,
                       X_dev,y_dev,cv=5,
                       scoring=['roc_auc','average_precision'],
                       return_estimator= True)
print(f"AUC score-{scores_under['test_roc_auc']}")
print(f"Average Precision score-{scores_under['test_average_precision']}")
print(f"Mean AUC score-{scores_under['test_roc_auc'].mean()}")
print(f"Mean Average Precision score-{scores_under['test_average_precision'].mean()}")


# In this case there are 5 models with the scores of each for various metrics as shown above. We select the model for further study based on the model that gave us the maximum AUC score. Given this assumption, we calculate and check to get the index of the best model from the given set of 5 models.

# In[12]:


under_best_index=np.argmax(scores_under['test_roc_auc'])
under_best_index


# **Q6.1. Perform random over sampling on the development set. What is the shape of your development features? How many positive and negative labels are there in your development set? (Please set random state as 42 when performing random over sampling)**

# In[13]:


# Your Code Here
ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X_dev, y_dev)
print(f'Shape of the development features matrix {X_over.shape}')
y_over= np.array(y_over)
non_zero_count = np.count_nonzero(y_over)
print(f'Out of which, count for class 0 is :{len(y_over)-non_zero_count}')
print(f'and count for class 1 is: {non_zero_count}')


# **Q6.2. Fit a default logistic regression model using over sampling. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing random over sampling)**

# In[14]:


pipe = imb_make_pipeline(ros,LogisticRegression())
scores_over = cross_validate(pipe,
                       X_dev,y_dev,cv=5,
                       scoring=['roc_auc','average_precision'],
                       return_estimator= True)
print(f"AUC score-{scores_over['test_roc_auc']}")
print(f"Average Precision score-{scores_over['test_average_precision']}")
print(f"Mean AUC score-{scores_over['test_roc_auc'].mean()}")
print(f"Mean Average Precision score-{scores_over['test_average_precision'].mean()}")


# In this case there are 5 models with the scores of each for various metrics as shown above. We select the model for further study based on the model that gave us the maximum AUC score. Given this assumption, we calculate and check to get the index of the best model from the given set of 5 models.

# In[15]:


over_best_index=np.argmax(scores_over['test_roc_auc'])
over_best_index


# **Q7.1. Perform Synthetic Minority Oversampling Technique (SMOTE) on the development set. What is the shape of your development features? How many positive and negative labels are there in your development set? (Please set random state as 42 when performing SMOTE)**

# In[16]:


# Your Code Here
smote = SMOTE(random_state = 42)
X_smote, y_smote = smote.fit_resample(X_dev, y_dev)
print(f'Shape of the development features matrix {X_smote.shape}')
y_smote= np.array(y_smote)
non_zero_count = np.count_nonzero(y_smote)
print(f'Out of which, count for class 0 is :{len(y_smote)-non_zero_count}')
print(f'and count for class 1 is: {non_zero_count}')


# **Q7.2. Fit a default logistic regression model using SMOTE. Print the AUC and average precision of 5-fold cross validation. (Please set random state as 42 when performing SMOTE)**

# In[17]:


# Your Code Here
pipe = imb_make_pipeline(smote,LogisticRegression())
scores_smote=cross_validate(pipe,
                       X_dev,y_dev,cv=5,
                       scoring=['roc_auc','average_precision'],
                       return_estimator= True)
print(f"AUC score-{scores_smote['test_roc_auc']}")
print(f"Average Precision score-{scores_smote['test_average_precision']}")
print(f"Mean AUC score-{scores_smote['test_roc_auc'].mean()}")
print(f"Mean Average Precision score-{scores_smote['test_average_precision'].mean()}")


# In this case there are 5 models with the scores of each for various metrics as shown above. We select the model for further study based on the model that gave us the maximum AUC score. Given this assumption, we calculate and check to get the index of the best model from the given set of 5 models.

# In[18]:


smote_best_index=np.argmax(scores_smote['test_roc_auc'])
smote_best_index


# **Q8. Plot confusion matrices on the test set for all four models above. Comment on your result.**

# In[19]:


# Your Code Here
y_pred=scores_logReg['estimator'][log_best_index].predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='g')
plt.title("Confusion matrix for vanilla Logistic Regression")
plt.show()

y_pred=scores_under['estimator'][under_best_index].predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='g')
plt.title("Confusion matrix for Logistic Regression after undersampling")
plt.show()


y_pred=scores_over['estimator'][over_best_index].predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='g')
plt.title("Confusion matrix for Logistic Regression after oversampling")
plt.show()


y_pred=scores_smote['estimator'][smote_best_index].predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='g')
plt.title("Confusion matrix for Logistic Regression after SMOTE")
plt.show()


# We notice that as we use sampling techniques such as oversampling, undersampling or SMOTE the correct prediction of the minority class increases from the vanilla logistic regression. But we also see an increase in False Positives which is a consequence of doing this.We also notice the False negative values decrease after sampling from 35 to 8. This is key for our problem of credit card fraud detection, where the number of false negatives should be low, i.e., model should reduce the number of times fraud is classified as "no fraud".
# At the same time we should also keep an eye on False positives aswell. For undersampled data the False positives are higher than that of oversampling and SMOTE.False positive results in card getting blocked for transactions that are not actually fraud.

# **Q9. Plot the ROC for all four models above in a single plot. Make sure to label the axes and legend. Comment on your result.**

# In[20]:


# Your Code Here
y_pred=scores_logReg['estimator'][log_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
plt.figure(figsize = (16,10))
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression_(AUC = %0.3f)" % roc_score)

y_pred=scores_under['estimator'][under_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression_UnderSampling(AUC = %0.3f)" % roc_score)

y_pred=scores_over['estimator'][over_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression_OverSampling(AUC = %0.3f)" % roc_score)

y_pred=scores_smote['estimator'][smote_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression_SMOTE(AUC = %0.3f)" % roc_score)
plt.title("ROC plots for various models")
plt.xlabel("False Positive Rate")
plt.legend()
plt.ylabel("True Positive Rate")


# We see that for lower values of False positive rate the Sampled versions of logistic regression perform better and for higher values the vanilla logistic regression does the better job.Hence, ROC curves do not give the full picture for imbalanced datasets and we need to look into the PR curves to get the best model.ROC curves may have high values even if it mis classifies the minority data sets and hence there is a need to look at PR curves.

# **Q10. Plot the precision-recall curve for all four models above in a single plot. Make sure to label the axes and legend. Comment on your result.**

# In[21]:


# Your Code Here
y_pred=scores_logReg['estimator'][log_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
plt.figure(figsize = (15,10))
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression_(AP = %0.3f)" % avg_score)


y_pred=scores_under['estimator'][under_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression_UnderSampling(AP = %0.3f)" % avg_score)

y_pred=scores_over['estimator'][over_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression_OverSampling(AP = %0.3f)" % avg_score)

y_pred=scores_smote['estimator'][smote_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression_SMOTE(AP = %0.3f)" % avg_score)

plt.title("PR curves plots for various models")
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")


# Average precision of models are high when it can handle prediction of minority class better without accidently marking too many negative samples as positives.
# In this case we see that for various sampling methods there is a increase in the recall, and since recall and precision are metrics that are opposing in this case, there will be a decrease precision.Hence we see that for sampling techniques the Average precision is lower than the vanilla logistic regression. In the case of undersampling we see that the value is less than all since it also has the highest number of false positives and this affects the precision a lot and hence it has the least average precision.

# **Q11. Adding class weights to a logistic regression model. Print the AUC and average precision of 5-fold cross validation. Also, plot its confusion matrix on test set.**

# In[22]:


# Your Code Here
scores_logReg_balanced= cross_validate(LogisticRegression(class_weight='balanced'),
                       X_dev,y_dev,cv=5,
                       scoring=['roc_auc','average_precision'],
                       return_estimator= True)
print(f"AUC score-{scores_logReg_balanced['test_roc_auc']}")
print(f"Average Precision score-{scores_logReg_balanced['test_average_precision']}")
print(f"Mean AUC score-{scores_logReg_balanced['test_roc_auc'].mean()}")
print(f"Mean Average Precision score-{scores_logReg_balanced['test_average_precision'].mean()}")


# In this case there are 5 models with the scores of each for various metrics as shown above. We select the model for further study based on the model that gave us the maximum AUC score. Given this assumption, we calculate and check to get the index of the best model from the given set of 5 models.

# In[23]:


log_balanced_best_index= np.argmax(scores_logReg_balanced['test_roc_auc'])
log_balanced_best_index


# In[24]:


y_pred=scores_logReg_balanced['estimator'][log_balanced_best_index].predict(X_test)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='g')
plt.title("Confusion matrix for balanced weight's Logistic Regression")
plt.show()


# **Q12. Plot the ROC and the precision-recall curve for default Logistic without any sampling method and this balanced Logistic model in two single plots. Make sure to label the axes and legend. Comment on your result.**

# In[25]:


# Your Code Here
y_pred=scores_logReg['estimator'][log_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
plt.figure(figsize = (16,10))
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression(AUC = %0.3f)" % roc_score)

y_pred=scores_logReg_balanced['estimator'][log_balanced_best_index].predict_proba(X_test)
fpr,tpr,_=roc_curve(y_test,y_pred[:,1])
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label="Logistic Regression_Balanced(AUC = %0.3f)" % roc_score)
plt.title("ROC plots for various models")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# In[26]:


y_pred=scores_logReg['estimator'][log_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
plt.figure(figsize = (16,10))
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression(AP = %0.3f)" % avg_score)

y_pred=scores_logReg_balanced['estimator'][log_balanced_best_index].predict_proba(X_test)
fpr,tpr,_=precision_recall_curve(y_test,y_pred[:,1])
avg_score=average_precision_score(y_test,y_pred[:,1])
plt.plot(fpr, tpr, label="Logistic Regression_Balanced Weights(AP = %0.3f)" % avg_score)

plt.title("PR plots for various models")
plt.legend()
plt.xlabel("Recall") 
plt.ylabel("Precision")


# In balanced weights case for logistic regression, we modify the loss function to account for the class weights. This results in having an effect similar to oversampling. Hence we again see that the PR curve helps us in understanding the problem better. As said previously, the recall should be increased for our problem of fraud detection and this in turn reduces our precision. Hence we see that the Average precision of logistic regression with balance class weights is lower than the vanilla regression. 

# ## Part 2: Unsupervised Learning
# 
# In this part, we will be applying unsupervised learning approaches to a problem in computational biology. Specifically, we will be analyzing single-cell genomic sequencing data. Single-cell genomics is a set of revolutionary new technologies which can profile the genome of a specimen (tissue, blood, etc.) at the resolution of individual cells. This increased granularity can help capture intercellular heterogeneity, key to better understanding and treating complex genetic diseases such as cancer and Alzheimer's. 
# 
# <img src="https://cdn.10xgenomics.com/image/upload/v1574196658/blog/singlecell-v.-bulk-image.png" width="800px"/>
# 
# <center>Source: 10xgenomics.com/blog/single-cell-rna-seq-an-introductory-overview-and-tools-for-getting-started</center>
# 
# A common challenge of genomic datasets is their high-dimensionality: a single observation (a cell, in the case of single-cell data) may have tens of thousands of gene expression features. Fortunately, biology offers a lot of structure - different genes work together in pathways and are co-regulated by gene regulatory networks. Unsupervised learning is widely used to discover this intrinsic structure and prepare the data for further analysis.

# ### Dataset: single-cell RNASeq of mouse brain cells

# We will be working with a single-cell RNASeq dataset of mouse brain cells. In the following gene expression matrix, each row represents a cell and each column represents a gene. Each entry in the matrix is a normalized gene expression count - a higher value means that the gene is expressed more in that cell. The dataset has been pre-processed using various quality control and normalization methods for single-cell data. 
# 
# Data source is on Coursework.

# In[27]:


cell_gene_counts_df = pd.read_csv('mouse_brain_cells_gene_counts.csv', index_col='cell')
cell_gene_counts_df


# Note the dimensionality - we have 1000 cells (observations) and 18,585 genes (features)!
# 
# We are also provided a metadata file with annotations for each cell (e.g. cell type, subtissue, mouse sex, etc.)

# In[28]:


cell_metadata_df = pd.read_csv('mouse_brain_cells_metadata.csv')
cell_metadata_df


# Different cell types

# In[29]:


cell_metadata_df['cell_ontology_class'].value_counts()


# Different subtissue types (parts of the brain)

# In[30]:


cell_metadata_df['subtissue'].value_counts()


# Our goal in this exercise is to use dimensionality reduction and clustering to visualize and better understand the high-dimensional gene expression matrix. We will use the following pipeline, which is common in single-cell analysis:
# 1. Use PCA to project the gene expression matrix to a lower-dimensional linear subspace.
# 2. Cluster the data using K-means on the first 20 principal components.
# 3. Use t-SNE to project the first 20 principal components onto two dimensions. Visualize the points and color by their clusters from (2).

# ## 1 PCA

# **Q1. Perform PCA and project the gene expression matrix onto its first 50 principal components. You may use `sklearn.decomposition.PCA`.**

# In[31]:


### Your code here
pca = PCA(n_components=50)
principalComponents = pca.fit_transform(cell_gene_counts_df)
principal_df = pd.DataFrame(data = principalComponents)
principal_df


# **Q2. Plot the cumulative proportion of variance explained as a function of the number of principal components. How much of the total variance in the dataset is explained by the first 20 principal components?**

# In[32]:


### Your code here
fig = plt.figure()
plt.bar(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_)
plt.xlabel('Principal Componenets')
plt.ylabel('Variance')
plt.title('Variance explained by each of the principal components')
plt.show()


# In[33]:


fig = plt.figure()
plt.plot(range(len(pca.explained_variance_ratio_)),np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal Componenets')
plt.ylabel('cumulative sum of variances')
plt.title('Cumulative sum of variances over principal components')
plt.show()
print(f'The first 20 principal components capture a total variance of {sum(pca.explained_variance_ratio_[0:20])*100} %')


# **Q3. For the first principal component, report the top 10 loadings (weights) and their corresponding gene names.** In other words, which 10 genes are weighted the most in the first principal component?

# In[34]:


### Your code here
weights= pca.components_[0,:]
top_10_genes=sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:10]
genes=cell_gene_counts_df.columns
top_10_genes=pd.DataFrame(data=[genes[top_10_genes],weights[top_10_genes]],index=['Genes','Weights'])
print(f'The top 10 gene names with the corresponding weights in first principal component are -')
top_10_genes.T


# **Q4. Plot the projection of the data onto the first two principal components using a scatter plot.**

# In[35]:


### Your code here
first_2_components=np.array(principal_df[[0,1]])
fig = plt.figure(figsize=(10,8))
sns.scatterplot(x=first_2_components[:,0],y=first_2_components[:,1],alpha=0.5)
plt.xlabel('Principal Componenet 1')
plt.ylabel('Principal Componenet 2')
plt.title('Visualization using first 2 components')
plt.show()


# **Q5. Now, use a small multiple of four scatter plots to make the same plot as above, but colored by four annotations in the metadata: cell_ontology_class, subtissue, mouse.sex, mouse.id. Include a legend for the labels.** For example, one of the plots should have points projected onto PC 1 and PC 2, colored by their cell_ontology_class.

# In[36]:


### Your code here
classes=['cell_ontology_class', 'subtissue', 'mouse.sex', 'mouse.id']
class_df=cell_metadata_df[classes]
cord_df = pd.DataFrame(data=first_2_components)
cord_df.columns = ['cord1', 'cord2']
plot_df = pd.concat([cord_df, class_df], axis=1, join='inner')
plot_df  


# In[37]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))

ax1.set_title(classes[0])
sns.scatterplot(x='cord1', y='cord2', data=plot_df, ax=ax1,hue=classes[0],alpha =0.5)

ax2.set_title(classes[1])
sns.scatterplot(x='cord1', y='cord2', data=plot_df, ax=ax2,hue=classes[1],alpha =0.5)


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))

ax1.set_title(classes[2])
sns.scatterplot(x='cord1', y='cord2', data=plot_df, ax=ax1,hue=classes[2],alpha =0.5)

ax2.set_title(classes[3])
sns.scatterplot(x='cord1', y='cord2', data=plot_df, ax=ax2,hue=classes[3],alpha =0.5)


# **Q6. Based on the plots above, the first two principal components correspond to which aspect of the cells? What is the intrinsic dimension that they are describing?**

# ### Your answer here
# 
# These two dimensions principal components describe the cell ontology class. They are able to describe the various classes of the cell ontology class such as astrocyte, aligodendrocyte etc.

# ## 2 K-means

# While the annotations provide high-level information on cell type (e.g. cell_ontology_class has 7 categories), we may also be interested in finding more granular subtypes of cells. To achieve this, we will use K-means clustering to find a large number of clusters in the gene expression dataset. Note that the original gene expression matrix had over 18,000 noisy features, which is not ideal for clustering. So, we will perform K-means clustering on the first 20 principal components of the dataset.

# **Q7. Implement a `kmeans` function which takes in a dataset `X` and a number of clusters `k`, and returns the cluster assignment for each point in `X`. You may NOT use sklearn for this implementation. Use lecture 6, slide 14 as a reference.**

# In[38]:


def kmeans(X, k, iters=10):
    '''Groups the points in X into k clusters using the K-means algorithm.

    Parameters
    ----------
    X : (m x n) data matrix
    k: number of clusters
    iters: number of iterations to run k-means loop

    Returns
    -------
    y: (m x 1) cluster assignment for each point in X
    '''
    ### Your code here
    #Randomly choosing Centroids 
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :] 
     
    #finding the distance between centroids and all the data points
    distances = cdist(X, centroids ,'euclidean') 
    points = np.array([np.argmin(i) for i in distances]) 

    for _ in range(iters): 
        centroids = []
        for idx in range(k):
            temp_cent = X[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) 
        distances = cdist(X, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points


# Before applying K-means on the gene expression data, we will test it on the following synthetic dataset to make sure that the implementation is working.

# In[39]:


np.random.seed(0)
x_1 = np.random.multivariate_normal(mean=[1, 2], cov=np.array([[0.8, 0.6], [0.6, 0.8]]), size=100)
x_2 = np.random.multivariate_normal(mean=[-2, -2], cov=np.array([[0.8, -0.4], [-0.4, 0.8]]), size=100)
x_3 = np.random.multivariate_normal(mean=[2, -2], cov=np.array([[0.4, 0], [0, 0.4]]), size=100)
X = np.vstack([x_1, x_2, x_3])

plt.figure(figsize=(8, 5))
sns.scatterplot(X[:, 0], X[:, 1], s=10,alpha =0.5)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)


# **Q8. Apply K-means with k=3 to the synthetic dataset above. Plot the points colored by their K-means cluster assignments to verify that your implementation is working.**

# In[40]:


### Your code here
classes=kmeans(X, 3, iters=100)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X[:, 0], y= X[:, 1],hue=classes,alpha =0.5)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.title('k-means classification for synthetic dataset')


# **Q9. Use K-means with k=20 to cluster the first 20 principal components of the gene expression data.**

# In[41]:


### Your code here
principal_20_df=principal_df.iloc[:,0:20]
principal_20_df


# In[42]:


classes=kmeans(np.array(principal_20_df), 20, iters=100)
plt.figure(figsize=(8, 5))
plt.bar(range(len(np.bincount(classes))),np.bincount(classes))
plt.xlabel('classes', fontsize=15)
plt.xticks(np.arange(0,20,1))
plt.ylabel('Number of points in that class', fontsize=15)
plt.title('Number of datapoints assigned to each class from class 0 to class 19')


# ## 3 t-SNE

# In this final section, we will visualize the data again using t-SNE - a non-linear dimensionality reduction algorithm. You can learn more about t-SNE in this interactive tutorial: https://distill.pub/2016/misread-tsne/.

# **Q10. Use t-SNE to reduce the first 20 principal components of the gene expression dataset to two dimensions. You may use `sklearn.manifold.TSNE`.** Note that it is recommended to first perform PCA before applying t-SNE to suppress noise and speed up computation.

# In[43]:


### Your code here
tsne = TSNE(n_components=2, verbose=1, random_state=123)
t_sne_2 = tsne.fit_transform(principal_20_df)


# **Q11. Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions.**

# In[44]:


cord_df = pd.DataFrame(data=t_sne_2)
plot_df = pd.concat([cord_df,pd.DataFrame(data=classes)], axis=1, join='inner')
plot_df.columns = ['cord1', 'cord2','class']
plot_df 


# In[45]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x='cord1', y='cord2',
    data=plot_df,
    legend="full",
    alpha=0.3
)
plt.title('20 principal components projected onto first two dimensions of t-SNE')


# **Q12. Plot the data (first 20 principal components) projected onto the first two t-SNE dimensions, with points colored by their cluster assignments from part 2.**

# In[46]:


### Your code here
plt.figure(figsize=(16,10))
sns.scatterplot(
    x='cord1', y='cord2',
    data=plot_df,
    hue='class',
    legend="full",
    alpha=0.7
)
plt.title("20 principal components projected onto first two dimensions of t-SNE and it's class assignments")


# **Q13. Why is there overlap between points in different clusters in the t-SNE plot above?**

# ### Your answer here
# In K-Means, all of the sample points are assigned to a unique cluster that is closest to them, and hence, there cannot be any overlap between different clusters. Had we been able to plot and color the 20 dimensional projections of PCA by their clusters, we would see no overlap in the 20-dimensional space. However, in our case, to visualize the results of K-Means, we're again projecting the projections of PCA (using PCA 20 components) to a 2-dimensional space, which will not preserve the mutual distance and positioning between these points and hence, if we were to use the same cluster labels as that of K-Means (which we ran on the projections of PCA), we are likely to see overlap. This explains the overlap between points in different clusters.

# These 20 clusters may correspond to various cell subtypes or cell states. They can be further investigated and mapped to known cell types based on their gene expressions (e.g. using the K-means cluster centers). The clusters may also be used in downstream analysis. For instance, we can monitor how the clusters evolve and interact with each other over time in response to a treatment.
