import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
application_data = pd.read_csv('application_data.csv')

# Step 2: Data Preprocessing
# No preprocessing steps mentioned

# Step 3: Exploratory Data Analysis (EDA)

# Correlation between loan types, amounts, purposes, and default risk
plt.figure(figsize=(10, 6))
sns.barplot(x='NAME_CONTRACT_TYPE', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Loan Type')
plt.xlabel('Loan Type')
plt.ylabel('Default Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='AMT_CREDIT', y='TARGET', data=application_data)
plt.title('Correlation between Loan Amount and Default Risk')
plt.xlabel('Loan Amount')
plt.ylabel('Default Rate')
plt.show()

# Impact of loan purpose on default risk
plt.figure(figsize=(12, 6))
sns.barplot(x='NAME_CASH_LOAN_PURPOSE', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Loan Purpose')
plt.xlabel('Loan Purpose')
plt.ylabel('Default Rate')
plt.xticks(rotation=90)
plt.show()

# Difference in approval rates between cash and revolving loans
plt.figure(figsize=(10, 6))
sns.barplot(x='NAME_CONTRACT_TYPE', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Loan Type')
plt.xlabel('Loan Type')
plt.ylabel('Default Rate')
plt.show()

# Impact of previous application outcomes on future default risk
plt.figure(figsize=(10, 6))
sns.barplot(x='NAME_CONTRACT_STATUS', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Previous Application Outcome')
plt.xlabel('Previous Application Outcome')
plt.ylabel('Default Rate')
plt.show()

# Correlation between credit risk and weekday or hour of loan application
plt.figure(figsize=(12, 6))
sns.barplot(x='WEEKDAY_APPR_PROCESS_START', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Weekday of Loan Application')
plt.xlabel('Weekday of Loan Application')
plt.ylabel('Default Rate')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='HOUR_APPR_PROCESS_START', y='TARGET', data=application_data, estimator=lambda x: sum(x) / len(x))
plt.title('Default Rate by Hour of Loan Application')
plt.xlabel('Hour of Loan Application')
plt.ylabel('Default Rate')
plt.show()
