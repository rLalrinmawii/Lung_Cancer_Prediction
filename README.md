# Lung Cancer Prediction using Logistic Regression

## Aim
To utilize logistic regression to predict lung cancer using a dataset of cancer patients, conducting the analysis with Python.

## About the Dataset
This dataset comprises lung cancer patient information including the following columns:
- **age**
- **gender**
- **air pollution exposure**
- **alcohol use**
- **dust allergy**
- **occupational hazards**
- **genetic risk**
- **chronic lung disease**
- **balanced diet**
- **obesity**
- **smoking status**
- **passive smoker status**
- **chest pain**
- **coughing of blood**
- **fatigue levels**
- **weight loss**
- **shortness of breath**
- **wheezing**
- **swallowing difficulty**
- **clubbing of finger nails**
- **frequent colds**
- **dry coughs**
- **snoring**

By analyzing this data, we aim to understand the factors contributing to lung cancer and optimize treatment strategies.

## Libraries
The project utilizes several Python libraries for data analysis, visualization, and machine learning tasks:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Matplotlib** and **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning algorithms, model evaluation metrics (mean squared error, accuracy, classification report, and confusion matrix).

## Exploratory Data Analysis

1. **Loading the Dataset**: Load the dataset into a Pandas DataFrame (`df`).

2. **Initial Inspection**: Display the first few rows to inspect the structure and content of the DataFrame.

3. **Data Structure**: Check for missing values and identify data types of each column.

4. **Descriptive Statistics**: Generate descriptive statistics for numerical columns.

5. **Missing Values**: Count the number of missing values in each column.

6. **Severity Level Analysis**: Analyze the 'Level' column, which represents the severity of lung cancer ('Low', 'Medium', 'High').

## Data Preprocessing

1. **Mapping Categorical Values**: Map 'Low', 'Medium', and 'High' in the 'Level' column to numerical equivalents 1, 2, and 3, respectively.

2. **Heatmap**: Generate a heatmap to depict the correlation among variables in the dataset.

## Applying Logistic Regression

1. **Defining Variables**: Use 'Level' as the response variable (`y`) and the remaining variables as explanatory variables (`X`).

2. **Train-Test Split**: Split the dataset into training (70%) and testing (30%) sets.

3. **Model Training**: Train the Logistic Regression model on the training set.

4. **Model Evaluation**:
   - Evaluate accuracy and mean squared error on the training data.
   - Compare predictions with actual values on the test set.
   - Compute predicted probabilities for class labels in the test dataset.
   - Create a DataFrame to display actual values, predicted values, and corresponding predicted probabilities for the first five samples.

## Model Evaluation

1. **Accuracy**:
   - Training accuracy: 92.57%
   - Testing accuracy: 90.33%

2. **Classification Report**: Provides precision, recall, and F1-score for each class label.

3. **Confusion Matrix**:
   - True Negatives (TN): Correctly predicted 'Low' severity.
   - True Positives (TP) for 'Medium' and 'High' severity.
   - False Positives (FP) and False Negatives (FN) indicating incorrect predictions between 'Low', 'Medium', and 'High' severity levels.

## Conclusion
The logistic regression model demonstrates promising performance in predicting lung cancer severity, particularly for 'High' severity cases. However, there are challenges in accurately classifying 'Low' and 'Medium' severity levels, indicating a need for further refinement. Continued model tuning and exploration of more sophisticated algorithms are recommended to enhance predictive accuracy and utility in clinical practice.

---

## Usage

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/rLalrinmawii/Lung_Cancer_Prediction.git
   cd Lung_Cancer_Prediction
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Navigate and open `project.ipynb` to run the analysis step-by-step.

---

I am open to collaboration and welcome contributions to improve this project. If you have any suggestions, ideas, or want to discuss potential enhancements, feel free to open an issue or submit a pull request. Let's work together to advance lung cancer prediction and make a meaningful impact in healthcare.

