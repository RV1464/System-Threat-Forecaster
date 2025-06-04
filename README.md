# System Threat Forecaster: A Machine Learning Approach
This repository contains the code and analysis for the "System Threat Forecaster" Kaggle competition. The project aims to build a machine learning model to predict system threats based on various system telemetry data.

## Kaggle Competition & Dataset
- Kaggle Competition: https://www.kaggle.com/competitions/System-Threat-Forecaster
- Train Dataset: The primary training data, train.csv, is part of the Kaggle competition dataset. It can be accessed directly from the competition page after accepting the rules.
##  Project Structure & Methodology
The notebook follows a standard machine learning workflow, organized into the following key sections:

## Importing modules
This section includes all necessary libraries for data manipulation, visualization, preprocessing, model building, and evaluation, such as pandas, numpy, matplotlib, seaborn, sklearn, and xgboost.

## Data Loading
The train.csv and test.csv datasets are loaded for analysis and model training. The initial structure and dimensions of the data are examined.

#### Train Data Shape: (100000, 76)
#### Test Data Shape: (10000, 75)

## Exploratory Data Analysis [EDA]
The EDA phase is critical for gaining deep insights into the dataset's characteristics, potential issues, and underlying patterns. This section systematically explores the data through:

- #### Descriptive Statistics: 
Basic statistical summaries of numerical features are generated using .describe() to understand central tendencies, dispersion, and range.
- #### Unique Values and Value Counts: 
For categorical columns, the number of unique entries and the frequency of each category are identified, which is vital for understanding feature cardinality and distribution.
- #### Distribution Analysis: 
Visualizations, likely histograms or density plots, are used to inspect the distribution of individual numerical features and the target variable, identifying imbalances or skewed data.
- #### Correlation Analysis: 
A correlation matrix is computed, and heatmaps are generated to visualize the relationships between all features, particularly focusing on how individual features correlate with the target variable. This helps in identifying potential multicollinearity and feature relevance.
- #### Pandas-profiling: 
A comprehensive and automated EDA report is generated using pandas-profiling. This tool provides a quick, detailed overview of the data, including exhaustive analysis of missing values, in-depth correlation analysis, and visualizations for each feature, streamlining the initial data understanding process. This step is instrumental in uncovering initial data quality issues and guiding subsequent preprocessing decisions.

## Handling Missing Data
Addressing missing values is a crucial preprocessing step to ensure data quality and model performance. In this project, missing data points are first identified using methods like df.isnull().sum(). Columns identified with a high percentage of missing values (e.g., those with over 80-90% nulls) are strategically dropped, as they provide insufficient information and could introduce noise. For remaining features with fewer missing values, appropriate imputation techniques are applied. Depending on the nature of the feature, this might involve imputing numerical columns with their mean or median, and categorical columns with their mode, to maintain data integrity without introducing bias.

### Categorical Feature Encoding
Since most machine learning algorithms require numerical input, categorical features are transformed into a suitable format. This project employs:

- #### One-Hot Encoding: 
Applied to nominal categorical features (where no inherent order exists) to convert them into a numerical representation. Each unique category becomes a new binary column, preventing the model from inferring a false ordinal relationship between categories.
- #### Label Encoding: 
Potentially used for ordinal categorical features where a natural order exists, converting categories into integer labels. Careful attention is paid to ensure consistent encoding schemes are applied across both the training and test datasets to prevent discrepancies during model inference.

### Numerical Feature Scaling
Numerical features often have varying scales, which can bias certain machine learning algorithms towards features with larger magnitudes. To standardize their range, numerical features are scaled using:

- #### StandardScaler: 
This transformer standardizes features by removing the mean and scaling to unit variance. It is particularly effective for algorithms that assume features are normally distributed or are sensitive to the absolute scale of the input features, such as logistic regression and support vector machines. This process helps ensure that all features contribute equally to the distance calculations or gradient descent steps.

## Feature Engineering
This crucial step involves creating new features from existing ones to potentially enhance the model's predictive power and capture more complex, non-linear patterns within the data. While specific examples are not detailed here, common techniques that might have been applied include:

- #### Interaction Terms: 
Combining two or more existing features (e.g., multiplying them) to capture their synergistic effects.
- #### Polynomial Features: 
Generating polynomial combinations of numerical features to model non-linear relationships.
- #### Aggregations: 
Creating summary statistics (e.g., counts, sums, averages) from related data points, if applicable to the dataset structure. The goal is to provide the model with a richer, more informative representation of the underlying data.

## Feature Selection
To reduce dimensionality, mitigate overfitting, improve model interpretability, and accelerate training times, various feature selection techniques are applied. The project explores:

- #### SelectKBest with f_classif: 
This method selects the 'k' best features based on univariate statistical tests, specifically the F-value for classification, which measures the linear dependency between feature and target.
- #### Recursive Feature Elimination (RFE) with LogisticRegression: 
RFE works by recursively training the model and removing the least important features, or those with the smallest coefficients, until the desired number of features is reached.
- #### SequentialFeatureSelector: 
This technique performs greedy feature selection (either forward selection, adding features one by one, or backward elimination, removing features one by one) by evaluating subsets of features based on cross-validation scores. The outcome of this phase is a refined set of features that are most relevant to the prediction task.

## Model Training and Evaluation
A diverse set of machine learning models is trained and rigorously evaluated to identify the most effective solution for threat forecasting. The dataset is typically split into training (75%), validation (15%), and test (15%) sets to ensure robust evaluation. Models trained include:

- #### Logistic Regression: 
A fundamental linear classification model serving as a strong baseline.
- #### Decision Tree Classifier: 
A non-linear model that partitions the feature space based on feature values.
- #### Random Forest Classifier: 
An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees, enhancing robustness and reducing overfitting.
- #### XGBoost Classifier: 
A highly optimized gradient boosting framework known for its speed and performance on tabular data, often achieving state-of-the-art results. Initial model evaluations are performed using accuracy_score and f1_score on the validation set to assess initial performance before extensive tuning.

## Hyperparameter Tuning
To fine-tune the performance of the selected models, hyperparameter optimization is performed. This involves systematically searching for the best combination of parameters that yield the highest model performance on unseen data, thereby preventing overfitting and improving generalization. The project utilizes:

- #### GridSearchCV: 
Exhaustively searches over a specified parameter grid, evaluating all possible combinations for the best fit.
- #### RandomizedSearchCV: 
Randomly samples a fixed number of parameter settings from a given distribution, which is more computationally efficient for large search spaces. For the XGBoost Classifier, hyperparameter tuning resulted in optimal parameters such as {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}, leading to an improved accuracy of 0.6232.

## Final Model and Submission
Based on the comprehensive evaluation and hyperparameter tuning results, the XGBoost Classifier is chosen as the final model due to its superior performance and robustness in predicting system threats. The chosen model is then used to generate predictions on the entirely unseen test dataset. These predictions are formatted into a submission.csv file, adhering to the Kaggle competition's required structure with id and target columns, ensuring it is ready for submission and evaluation on the public and private leaderboards.
