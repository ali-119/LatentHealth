import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')



# Reading Files
demographic_file = pd.read_csv(demographic.csv)
diet_file = pd.read_csv(diet.csv)
labs_file = pd.read_csv(labs.csv)
examination_file = pd.read_csv(examination.csv)
medications_file = pd.read_csv(medications.csv, encoding='latin1')
questionnaire_file = pd.read_csv(questionnaire.csv)



# sort Data
demographic_data = demographic_file[["SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDAGEMN", "RIDRETH1", "RIDRETH3",
                                    "RIDEXMON", "RIDEXAGM", "DMQMILIZ", "DMQADFC", "DMDBORN4", "DMDCITZN", "DMDYRSUS", "DMDEDUC2",
                                     "DMDEDUC3", "DMDMARTL", "RIDEXPRG", "SIALANG", "SIAPROXY", "SIAINTRP", "FIALANG", "FIAPROXY",
                                      "FIAINTRP", "MIALANG", "MIAPROXY", "MIAINTRP", "AIALANGA", "DMDHHSIZ", "DMDFMSIZ", "DMDHHSZA",
                                       "DMDHHSZB", "DMDHHSZE", "DMDHRGND", "DMDHRAGE", "DMDHRBR4", "DMDHREDU", "DMDHRMAR", "DMDHSEDU",
                                        "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "INDHHIN2", "INDFMIN2", "INDFMPIR"]]

diet_data = diet_file[["SEQN", "WTDRD1", "WTDR2D", "DRQSDIET", "DRQSDT4", "DRQSDT6", "DRQSDT7", "DRQSDT9",
                        "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT"]]

labs_data = labs_file[["SEQN", "LBXGH", "LBXIN", "LBXTR", "LBXTC", "LBDLDL", "LBXHGB"]]

examination_data = examination_file[["SEQN", "BMXWT", "BMXHT", "BMXBMI"]]

medications_data = medications_file[["SEQN", "RXDDRUG", "RXDDRGID", "RXDUSE"]]

questionnaire_data = questionnaire_file[["SEQN", "DIQ010", "DID040", "DIQ160", "DIQ170", "DIQ070", "DIQ080", "PAQ605", "PAQ620", "PAD615", 
                                        "PAQ650", "PAQ665", "SMQ020", "SMQ040", "ALQ101", "ALQ130", "SLD010H", "MCQ160A"]]

df = demographic_data \
    .merge(diet_data, on="SEQN", how="left") \
    .merge(labs_data, on="SEQN", how="left") \
    .merge(examination_data, on="SEQN", how="left") \
    .merge(questionnaire_data, on="SEQN", how="left") \
    .merge(medications_data, on="SEQN", how="left")



# Cleaning Data
demographic_data_null = demographic_data.isna().sum()
print(demographic_data_null[demographic_data_null > 0])

diet_data_null = diet_data.isna().sum()
print(diet_data_null[diet_data_null > 0])

labs_data_null = labs_data.isna().sum()
print(labs_data_null[labs_data_null > 0])

examination_data_null = examination_data.isna().sum()
print(examination_data_null[examination_data_null > 0])

medications_data_null = medications_data.isna().sum()
print(medications_data_null[medications_data_null > 0])

questionnaire_data_null = questionnaire_data.isna().sum()
print(questionnaire_data_null[questionnaire_data_null > 0])

df.loc[df['SLD010H'] >= 77, 'SLD010H'] = np.nan

df = df.drop(['RIDAGEMN', 'DMQADFC', 'RIDEXAGM', 'DMQMILIZ', 'DMDYRSUS', 'DMDEDUC3', 'RIDEXPRG', 'MIALANG', 'MIAPROXY', 'MIAINTRP', 'AIALANGA',
            'DMDHSEDU', 'WTDR2D', 'DRQSDT4', 'DRQSDT6', 'DRQSDT7', 'DRQSDT9', 'LBDLDL', 'DID040', 'DIQ080', 'PAD615', 'PAQ650', 'PAQ665'], axis=1)

df["DIQ170"] = df["DIQ170"].fillna(0)

df["DMDEDUC2"] = df["DMDEDUC2"].fillna(9)  # Unknown
df["DMDMARTL"] = df["DMDMARTL"].fillna(9)

df["DRQSDIET"] = df["DRQSDIET"].fillna(2)

df["DIQ010"] = df["DIQ010"].fillna(2)  # 2 = No
df["DIQ160"] = df["DIQ160"].fillna(2)
df["DIQ070"] = df["DIQ070"].fillna(2)
df["MCQ160A"] = df["MCQ160A"].fillna(2)
activity_cols = ["PAQ605","PAQ620"]
df[activity_cols] = df[activity_cols].fillna(0)
df["SMQ020"] = df["SMQ020"].fillna(2)
df["SMQ040"] = df["SMQ040"].fillna(3)  # not current smoker
df["ALQ101"] = df["ALQ101"].fillna(2)
df["ALQ130"] = df["ALQ130"].fillna(0)

num_cols = ['INDFMPIR','DR1TKCAL','DR1TSUGR','DR1TFIBE','DR1TSFAT', 'LBXGH','LBXIN','LBXTR','LBXTC','LBXHGB','BMXWT','BMXHT','BMXBMI','SLD010H', 'DR1TTFAT',
            'RIDEXMON', 'DMDCITZN', 'SIAPROXY', 'FIALANG', 'FIAPROXY', 'FIAINTRP', 'DMDHRBR4', 'DMDHREDU', 'DMDHRMAR', 'INDHHIN2', 'INDFMIN2', 'WTDRD1', 'DR1TCARB']

df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

df['any_medication'] = df['RXDDRUG'].notna().astype(int)
df = df.drop(columns=['RXDDRUG','RXDDRGID'])

df = df[df["DIQ010"].isin([1, 2])]
df["diabetes"] = (df["DIQ010"] == 1).astype(int)


df["diabetes"] = (
    (df["DIQ010"].isin([1, 2])) &
    (df["DIQ070"].isin([1, 2])) &
    (df["DIQ160"].isin([1, 2]))
     ).astype(int)
df = df.drop(columns = ["DIQ010", "DIQ070", "DIQ160"])

df["physically_active"] = (
    (df["PAQ605"] == 1) |
    (df["PAQ620"] == 1)
).astype(int)
df = df.drop(columns=["PAQ605","PAQ620"])

df["smoking_status"] = np.select(
    [
        df["SMQ040"] == 1,        # current smoker
        df["SMQ020"] == 1,        # former smoker
    ],
    [2, 1],
    default=0                   # never smoker
)
df = df.drop(columns=["SMQ020","SMQ040"])

df["alcohol_use"] = np.where(
    df["ALQ101"] == 2,  # No
    0,
    df["ALQ130"]
)
df = df.drop(columns=["ALQ101","ALQ130"])

df["sugar_ratio"] = df["DR1TSUGR"] / df["DR1TCARB"]
df["fiber_density"] = df["DR1TFIBE"] / df["DR1TKCAL"]
df["satfat_ratio"] = df["DR1TSFAT"] / df["DR1TTFAT"]
df = df.drop(columns=["DR1TCARB", "DR1TSUGR", "DR1TFIBE","DR1TTFAT", "DR1TSFAT"])

df.loc[df['alcohol_use'] >= 777, 'alcohol_use'] = np.nan
df['alcohol_use'] = df['alcohol_use'].fillna(0)
df = df[df['alcohol_use'] < 11.0]

df['satfat_ratio'] = df['satfat_ratio'].fillna(df['satfat_ratio'].mean())



# Virtualization
# Seeing the distribution of diabetes across alcohol consumption groups
sns.countplot(x=df['alcohol_use'], hue=df['diabetes'], palette='viridis')
plt.yticks(range(0, 10001, 1000))
plt.xlabel('Alcohol Consumption')
plt.show()

print(df['alcohol_use'].value_counts())

# Seeing the distribution of diabetes in sleep group
sns.countplot(x=df['SLD010H'], hue=df['diabetes'], palette='viridis')
plt.xlabel('Sleep')
plt.show()

# Seeing the distribution of diabetes across smoking groups
sns.countplot(x=df['smoking_status'], hue=df['diabetes'], palette='viridis')
plt.xlabel('Smoking')
plt.show()

# Seeing the distribution of diabetes in physically active groups
sns.countplot(x=df['physically_active'], hue=df['diabetes'], palette='viridis')
plt.xlabel('Physically Active')
plt.show()



# Data partitioning and training and testing
standard_scale = StandardScaler()
standard_scale.fit_transform(df)

X = df.drop(columns=['diabetes', 'SEQN'], axis=1)
Y = df['diabetes']

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=119, test_size=0.25)

standard_scale = StandardScaler()
x_train_scaled = standard_scale.fit_transform(x_train)
x_test_scaled = standard_scale.transform(x_test)



# creating basic model
def basic_model(model):
    model = model
    model_fit = model.fit(x_train_scaled, y_train)
    model_pred = model_fit.predict(x_test_scaled)
    return f"""for {model}
Accuracy Score: {accuracy_score(y_test, model_pred)}
Classification Report:\n{classification_report(y_test, model_pred)}\n
"""
print(basic_model(LogisticRegression()))
print(basic_model(SVC()))
print(basic_model(DecisionTreeClassifier()))
print(basic_model(RandomForestClassifier()))
print(basic_model(GradientBoostingClassifier()))
print(basic_model(KNeighborsClassifier()))



# Choosing best model
log_param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
}
svc_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5]
}
decision_param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [7, 9],
    'min_samples_split': [4, 6],
}
random_param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 4, 6, 8],
}
gradiant_param_grid = {
    'loss': ['log_loss', 'deviance', 'exponential'],
    'learning_rate': [0.2, 0.4, 0.6],
    'n_estimators': [50, 100, 150],
}
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'] ,
    'algorithm': ['auto', 'kd_tree', 'brute'],
}

li = []

model = [LogisticRegression(max_iter=1000, random_state=119), SVC(random_state=119),
        DecisionTreeClassifier(random_state=119), RandomForestClassifier(random_state=119),
        GradientBoostingClassifier(random_state=119), KNeighborsClassifier()]
param_grid = [log_param_grid, svc_param_grid, decision_param_grid, random_param_grid, gradiant_param_grid, knn_param_grid]
for m, p in zip(model, param_grid):
    grid = GridSearchCV(estimator=m, param_grid=p, verbose=1, cv=3)
    grid.fit(x_train_scaled, y_train)
    grid_pred = grid.predict(x_test_scaled)
    results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    li.append(results)
    
    print(f"""for {m} 
        Accuracy Score: {accuracy_score(y_test, grid_pred)} 
        Classification Report:\n{classification_report(y_test, grid_pred)}\n 
        Best Params: {grid.best_params_}\n""")

print(li)

'''
Accuracy ~ 99.95% every models
0 = 6
1 = 4842
-----> Overfitting
'''



# Choosing best mode for KNN by elbow method
test_error_rates = []
for k in range(1,31):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train_scaled, y_train)
    y_pred_test = knn_model.predict(x_test_scaled)

    test_error = 1 - accuracy_score(y_test, y_pred_test)
    test_error_rates.append(test_error_rates)

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(range(1,31), test_error_rates, label = 'test error')
plt.legend()
plt.ylabel("test_error_rates")
plt.xlabel("k value")
plt.grid()
plt.show()



# Using Clustering
## Scale data for Clustering
minmax_scale = MinMaxScaler()
mm_scale = minmax_scale.fit_transform(df)

df_scaled = pd.DataFrame(data=mm_scale, columns=df.columns)



## Elbow method for Clustering
### DBSCAN
out_percent_list = []
for ms in [3, 5, 10]:
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        db_model = DBSCAN(eps=eps, min_samples=ms)
        db_labels = db_model.fit_predict(df_scaled)

        percent_of_outlier = 100 * np.sum(db_labels == -1) / len(db_labels)
        out_percent_list.append(percent_of_outlier)
        
        print(f"For eps {eps}, min sample {ms}")
        print(f"Number Outliers: {np.sum(db_labels == -1)}")
        print(f"Number of clusters: {len(np.unique(db_labels[db_labels != -1]))}\n")

plt.plot(range(1, 16, 1), out_percent_list, marker='o', markerfacecolor='darkblue', markersize=4)
plt.yticks(range(0, 101, 20))
plt.xticks(range(1, 16, 1))
plt.ylabel("Percentage of Points Classified as Outlier")
plt.xlabel("Value")
plt.title("Epsilon Value to Percent of Outlier")
plt.show()

'''
best:
    For eps 1.0, min sample 3
    Number Outliers: 2268
    Number of clusters: 857
'''

# ### KMeans
ss = []
inertia = []
for k in range(1, 21):
    model = KMeans(n_clusters=k, random_state=101)
    model = model.fit(df_scaled)

    inertia.append(model.inertia_)
    if k > 1:
        score = silhouette_score(df_scaled, model.labels_)
        ss.append(score)

plt.plot(range(1, 21), inertia, 'o--')
plt.xticks(range(1,21))
plt.grid()
plt.show()

'''
best:
    n_clusters = 10
'''

### PCA
explain_variance = []
for components in range(1, 31):
    pca_model = PCA(n_components=components)
    pca = pca_model.fit(df_scaled)

    explain_variance.append(np.sum(pca.explained_variance_ratio_))

plt.plot(range(1, 31), explain_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Variance explain")
plt.grid()
plt.show()

'''
best:
    n_components = 15
'''



# Finall Model
model = KMeans(n_clusters=10, random_state=119)
model_pred = model.fit_predict(X)

df_scaled = df_scaled[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'DMDCITZN', 'DMDBORN4']]

df_scaled = df_scaled.rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'RIDRETH1': 'race',
    'RIDRETH3': 'Detailed_breed',
    'DMDEDUC2': 'education',
    'DMDMARTL': 'marital_status',
    'DMDCITZN': 'citizenship',
    'DMDBORN4': 'place_of_birth',
})

df_scaled['Cluster'] = model_pred

cluster_summary = df_scaled.groupby('Cluster').mean()
cluster_summary = df_scaled.groupby('Cluster').mean(numeric_only=True)

cluster_df = pd.DataFrame(cluster_summary.iloc[0:10, :])

cluster_df.to_csv('cluster_summary.csv', index=False, sep='\t')

cluster_summary.drop('SEQN', axis=1).T.iloc[0:10, :].plot(kind='bar', figsize=(12,6))
plt.xticks(rotation=0)
plt.show()
