# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Updated to avoid deprecation
sns.set_palette("muted")

# Load the dataset
df = pd.read_csv("C:/Users/sainh/Downloads/Accidental_Drug_Related_Deaths_2012-2023.csv")

# Data Cleaning
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Fill missing values for categorical columns with 'Unknown'
categorical_cols = ['Sex', 'Race', 'Ethnicity', 'Residence City', 'Death City', 'Location']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# Convert drug columns to boolean (Y -> True, else -> False)
drug_columns = ['Heroin', 'Cocaine', 'Fentanyl', 'Fentanyl Analogue', 'Oxycodone', 'Oxymorphone',
                'Ethanol', 'Hydrocodone', 'Benzodiazepine', 'Methadone', 'Meth/Amphetamine',
                'Amphet', 'Tramad', 'Hydromorphone', 'Morphine (Not Heroin)', 'Xylazine',
                'Gabapentin', 'Opiate NOS', 'Any Opioid']
for col in drug_columns:
    df[col] = df[col].apply(lambda x: True if x == 'Y' else False)

# Objective 1: Temporal Trends
print("Objective 1: Temporal Trends")
df['Year'] = df['Date'].dt.year
yearly_deaths = df.groupby('Year').size()

plt.figure(figsize=(10, 6))
yearly_deaths.plot(kind='line', marker='o')
plt.title('Drug-Related Deaths by Year (2012-2023)')  # Fixed invalid character
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True)
plt.savefig('yearly_deaths_trend.png')
plt.show()

# Objective 2: Drug Prevalence
print("\nObjective 2: Drug Prevalence")
drug_counts = df[drug_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
drug_counts.plot(kind='bar')
plt.title('Prevalence of Substances in Drug-Related Deaths')
plt.xlabel('Substance')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('drug_prevalence.png')
plt.show()

# Objective 3: Demographic Analysis
print("\nObjective 3: Demographic Analysis")
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution of Drug-Related Deaths')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.show()

# Sex Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df)
plt.title('Drug-Related Deaths by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('sex_distribution.png')
plt.show()

# Race Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Race', data=df, order=df['Race'].value_counts().index)
plt.title('Drug-Related Deaths by Race')
plt.xlabel('Count')
plt.ylabel('Race')
plt.savefig('race_distribution.png')
plt.show()

# Objective 4: Geographic Distribution
print("\nObjective 4: Geographic Distribution")
top_cities = df['Death City'].value_counts().head(10)
plt.figure(figsize=(12, 8))
top_cities.plot(kind='bar')
plt.title('Top 10 Cities by Number of Drug-Related Deaths')
plt.xlabel('City')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('geographic_distribution.png')
plt.show()

# Objective 5: Location of Death
print("\nObjective 5: Location of Death")
plt.figure(figsize=(10, 6))
sns.countplot(y='Location', data=df, order=df['Location'].value_counts().index)
plt.title('Common Locations of Drug-Related Deaths')
plt.xlabel('Count')
plt.ylabel('Location')
plt.savefig('location_of_death.png')
plt.show()

# Objective 6: Substance Co-Occurrence
print("\nObjective 6: Substance Co-Occurrence")
# Correlation matrix for drug columns
drug_corr = df[drug_columns].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(drug_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Substances in Drug-Related Deaths')
plt.savefig('drug_correlation_heatmap.png')
plt.show()

# Statistical Summary
print("\nStatistical Summary:")
print(f"Total number of deaths: {len(df)}")
print(f"Average age: {df['Age'].mean():.1f} years")
print(f"Most common drug: {drug_counts.idxmax()} ({drug_counts.max()} cases)")
print(f"Most common death city: {df['Death City'].value_counts().idxmax()} ({df['Death City'].value_counts().max()} deaths)")
print(f"Most common location: {df['Location'].value_counts().idxmax()} ({df['Location'].value_counts().max()} deaths)")

# Save the cleaned dataset
df.to_csv('cleaned_drug_deaths.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_drug_deaths.csv'")