# Unsupervised-Health-Data-Analysis-using-NHANES-Dataset
A machine learning project focused on **unsupervised learning**, **pattern discovery**, and **population-level health analysis** using the **NHANES (National Health and Nutrition Examination Survey)** dataset.

<p align="center">
  <img src="https://img.shields.io/badge/ML-Unsupervised_Learning-green">
  <img src="https://img.shields.io/badge/Clustering-KMeans%20%7C%20DBSCAN%20%7C%20Hierarchical-blue">
  <img src="https://img.shields.io/badge/Dimensionality_Reduction-PCA-orange">
  <img src="https://img.shields.io/badge/Dataset-NHANES-purple">
  <img src="https://img.shields.io/badge/Language-Python-yellow">
</p>

------

# Overview
This project explores **hidden health-related patterns** in large-scale population data using **unsupervised machine learning techniques**.

Instead of predicting a predefined target, the goal is to:
- Discover natural population subgroups  
- Identify latent health profiles  
- Analyze relationships between clinical, demographic, and lifestyle variables  
- Compare clustering behaviors across multiple algorithms  

The project emphasizes **interpretability**, **data-driven insight**, and **real-world complexity** rather than model accuracy alone.

------

# Dataset
- **Name:** NHANES (National Health and Nutrition Examination Survey)
- **Source:** [Data set](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey)
- **Type:** Real-world health survey data
- **Scale:** Large, multi-table, heterogeneous dataset

## Data Characteristics
- Demographic features (age, gender, ethnicity, etc.)
- Clinical measurements
- Laboratory results
- Lifestyle and health indicators
- Missing values and mixed distributions (realistic noise)

> This dataset reflects real population health conditions and is significantly more complex than typical educational datasets.

------

# Project Objectives
- Perform structured preprocessing on complex health data  
- Reduce dimensionality for better visualization and analysis  
- Apply and compare multiple clustering algorithms  
- Interpret discovered clusters from a health perspective  
- Build a reusable framework for unsupervised health analysis  

------

# Project Workflow

## 1) Data Preparation & Cleaning

- Selected relevant features from the NHANES dataset
- Handled missing values
- Removed non-informative or redundant attributes
- Prepared a clean analytical dataset

> The focus was on preserving **real-world data structure** rather than over-sanitizing the dataset.

## Data Preprocessing & Feature Engineering

Key preprocessing steps include:

- Merging multiple NHANES tables using `SEQN`
- Handling missing values using:
  - Domain-driven filling (categorical logic)
  - Mean imputation for numerical features
    
- Removing survey-specific noise codes (e.g. 777, 99)
- Feature engineering:
  - Sugar-to-carb ratio
  - Fiber density
  - Saturated fat ratio
  - Physical activity indicator
  - Smoking status
  - Alcohol consumption normalization
    
- Target construction:
  - Diabetes label derived from multiple questionnaire variables

-----

## 2) Feature Scaling

Used standardization to normalize feature distributions:

- `StandardScaler()`
- `MinMaxScaler()`
  
> This step ensures fair contribution of all features to distance-based models.

-----

## Supervised Learning (Classification)

Several models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVC)
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

-----

## 3) Dimensionality Reduction (PCA)

Applied **Principal Component Analysis (PCA)** to:

- Reduce high-dimensional feature space  
- Enable visualization in 2D and 3D  
- Capture dominant variance patterns  

PCA was used **for analysis and visualization**, not as a predictive model.

-----

## 4) Clustering Algorithms

Multiple clustering approaches were applied and compared to explore different structural perspectives of the data.

### KMeans Clustering

- Explored different numbers of clusters
- Used the elbow method to evaluate inertia
- Provided a baseline partitioning of the population

### DBSCAN (Density-Based Clustering)

- Detected dense population groups
- Identified outliers and rare health profiles
- Demonstrated sensitivity to data density and scaling

-----

## 5) Cluster Analysis & Interpretation

- Cluster assignments were analyzed statistically  
- Summary tables were generated to compare cluster-level characteristics  
- Differences between clusters were interpreted in a health-related context  

> Interpretation focuses on **patterns**, not medical diagnosis.

-----

## Results Summary

- Distinct population subgroups emerged without supervision
- Different clustering algorithms revealed complementary perspectives
- Density-based methods highlighted rare and extreme cases

> These results demonstrate that **unsupervised learning can extract meaningful structure from complex health data**.

-----

## Key Takeaways

- Real-world health data is noisy, incomplete, and non-trivial  
- Unsupervised learning is powerful for exploratory analysis  
- No single clustering method is universally optimal  
- Interpretation is as important as algorithm selection  
- PCA is a critical tool for understanding high-dimensional structure  

-----

## Limitations

- No causal inference or medical diagnosis  
- Cluster interpretation depends on selected features  
- NHANES dataset complexity limits full coverage in a single project  

> These limitations are intentional and reflect real data science constraints.

-----

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
  - `PCA`
  - `KMeans`
  - `DBSCAN`
  - `StandardScaler`
  - `DecisionTreeClassifier`
  - `KNeighborsClassifier`
  - `GradientBoostingClassifier`
  - `RandomForestClassifier`
  - `MinMaxScaler`
  - `GridSearchCV`
  - `...`

------

# How to Run

## Clone the repository:
[GitHub Repository](https://github.com/ali-119/Unsupervised-Health-Data-Analysis-using-NHANES-Dataset)

## Install dependencies:
```python
pip install -r requirements.txt
```
- requirements.txt → [File](https://github.com/ali-119/Unsupervised-Health-Data-Analysis-using-NHANES-Dataset/blob/main/requirements.txt)

or directly:
```python
pip install numpy pandas seaborn matplotlib scikit-learn
```
> Run the script to generate all visualizations.

------

## Cluster Analysis Results

Based on KMeans clustering with 10 groups, patients were segmented into distinct risk profiles:

| Cluster | Patient Group | Characteristics |
|---------|---------------|-----------------|
| 0 | Older Low-Education Males | Older age (65%), Predominantly male (63%), Low education level, Single/never married, US-born, Potential social isolation risk |
| 1 | Middle-Aged Immigrant Males | Middle age (59%), Male majority (56%), Medium education, Married, Foreign-born population, Cultural adaptation factors |
| 2 | Older Low-Education Females | Older age (62%), Predominantly female (43%), Low education, Single/unmarried, US-born, Higher vulnerability to chronic conditions |
| 3 | Elderly Males with Low SES | Elderly population (68%), Male majority (57%), Low education, Single/unmarried, US-born, High-risk demographic for diabetes complications |
| 4 | Young Educated Women | Youngest group (41%), Female majority (44%), Highest education level, Married, Foreign-born, Health-conscious demographic |
| 5 | Elderly Low-Education Population | Oldest group (72%), Gender-balanced, Very low education, Single/unmarried, US-born, Highest risk for health complications |
| 6 | Older Males with Moderate SES | Older age (64%), Male majority (65%), Medium education, Single, US-born, Moderate diabetes risk profile |
| 7 | Educated Married Immigrants | Middle age (48%), Gender-balanced, Highest education and marriage rate, Foreign-born, Likely higher health literacy |
| 8 | Middle-Aged Male Immigrants | Middle age (56%), Male majority (59%), Low-medium education, Single, Some foreign-born, Mixed risk factors |
| 9 | Elderly with Medium Education | Elderly (69%), Gender-balanced, Medium education, Single, US-born, Moderate-to-high risk for age-related conditions |

------

## Key Findings

### High-Risk Clusters
**Clusters 3, 5, and 9** represent the highest-risk patient populations:
- **Advanced age** (68-72% normalized age score)
- **Low educational attainment** limiting health literacy
- **Social isolation** (predominantly single/unmarried)
- **US-born** suggesting longer exposure to Western dietary patterns

These groups require **intensive diabetes screening**, **patient education programs**, and **social support interventions**.

### Protected Clusters
**Clusters 4 and 7** show protective characteristics:
- **Younger age** or middle-aged
- **Higher education levels** (53-59% education score)
- **Married status** providing social support
- **Foreign-born** potentially maintaining healthier traditional diets

These groups demonstrate better **health-seeking behaviors** and **preventive care engagement**.

### Gender-Specific Patterns
- **Male-dominated clusters** (0, 1, 3, 6, 8): Higher prevalence in older age groups
- **Female-dominated clusters** (2, 4): Split between elderly low-SES and young educated groups
- **Gender-balanced clusters** (5, 7, 9): Mixed demographics with varying education levels

### Socioeconomic Disparities
Clear stratification emerged across education levels:
- **Low education** (clusters 0, 2, 3, 5, 8): Associated with older age and single status
- **Medium education** (clusters 1, 6, 9): Mixed risk profiles
- **High education** (clusters 4, 7): Younger age, married, better health outcomes

------

## Clinical Implications

### Targeted Interventions
1. **Clusters 3, 5, 9**: Prioritize aggressive diabetes screening, medication adherence programs, and geriatric care coordination
2. **Clusters 0, 2**: Focus on social support services and simplified health education materials
3. **Clusters 4, 7**: Leverage as peer educators and community health advocates
4. **Clusters 1, 8**: Provide culturally-adapted health materials in multiple languages

### Resource Allocation
Healthcare systems should allocate:
- **70% of preventive resources** to high-risk clusters (0, 2, 3, 5, 9)
- **20% to moderate-risk** clusters (1, 6, 8)
- **10% to low-risk clusters** (4, 7) for maintenance and prevention

### Policy Recommendations
- Implement **age-stratified screening protocols** (65+ priority)
- Develop **education-level appropriate** diabetes education materials
- Create **culturally sensitive programs** for immigrant populations
- Establish **social support networks** for single/isolated patients

------

## Cluster Visualization

<img width="1200" height="600" alt="Figure" src="https://github.com/user-attachments/assets/6146420e-b91d-4a91-a27b-da1355b0d2bc" />

------

## Clustering results file

- cluster_summary.csv → [file](https://github.com/ali-119/Unsupervised-Health-Data-Analysis-using-NHANES-Dataset/blob/main/cluster_summary.csv)

------

## Conclusion

The clustering analysis successfully identified **10 distinct patient subgroups** with varying diabetes risk profiles. The most vulnerable populations are characterized by:
- Advanced age (>65 years)
- Low educational attainment
- Social isolation (single/unmarried status)
- US nativity with likely Western dietary exposure

This segmentation enables **precision public health interventions**, allowing healthcare providers to:
- Allocate resources efficiently to high-risk groups
- Design tailored prevention programs based on demographic characteristics
- Improve health equity by addressing socioeconomic disparities
- Optimize screening schedules and follow-up protocols

> The identification of protected clusters (4 and 7) provides valuable insights into **protective factors** that can inform **population-level prevention strategies**. Future research should investigate > behavioral and lifestyle differences across these clusters to develop evidence-based interventions for diabetes prevention and management.

------

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)
