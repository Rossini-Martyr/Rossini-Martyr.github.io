---
layout: post
title: EQAO School Performance Segmentation Project
image: "/posts/eqao-k-means-img.png"
tags: [Segmentation Analysis, PCA, Python]
---

## Project Overview

EQAO is an evidence-based research-informed organization in Ontario, Canada, that is focused on empowering educators, parents, guardians, stakeholders and the public at large with the insights and information needed to support student learning and improve student outcomes. To this end, they administer standardized tests to Grade 3, 6, and 9 in order to evaluate whether students at individual schools are performing to the provincial standard in the following subjects: Reading, Writing, Mathematics.

The main purpose of this project is to analyze the 2023-2024 Grade 6 EQAO results to find clusters of similarly performing schools. This information could be used for numerous purposes, e.g:

- Find schools that are outperforming their peers in a similar demographic such that their strategies can be studied and adopted
- Identify segments of schools that are poor in a particular area (literacy or maths, e.g.) and use this for targeted funding initiatives
- Examining schools to others with similar demographics within their cluster can allow for a fairer comparison instead of comparing to schools across the province

For clustering analysis, a K-Means model was used to segment the Ontario schools. By analyzing the Grade 6 student performance, the model identified 3 clusters of schools, briefly summarized as follows:

- High-performing English schools with strong performance in English, Reading and Writing.
- English schools with a poor Mathematics grade on average
- French schools

## Concept Overview

**K-Means Clustering** is an unsupervised machine learning algorithm used to discover naturally occurring groups within a dataset. Unlike supervised learning which requires the data to be labelled, K-Means looks through the data to find patterns that aren't immediately obvious.

**Methodology**:

- Define $k$ (the number of clusters you want to be created). The algorithm randomly places $k$ points, called centroids, into the data space.

- Assignment: Every data point is assigned to the nearest centroid based on its distance, forming $k$ clusters.

- Optimization: The centroids move to the center of their new groups, and the process repeats until the clusters are stable and the total variance within each group is minimized.

## Data Overview and Preparation

**Data Overview**

The data used for this project was made available through the Ministry of Education as reported by schools, school boards, EQAO and Statistics Canada. This data summarizes the exam results by school and board, combined with select demographic data on each school. The overall datasets can be found at the following link: [School Information and Student Demographics](https://data.ontario.ca/dataset/school-information-and-student-demographics)

The main dataset used was the 2023-2024 dataset. This dataset includes the all data related to the performance of the Grade 3, 6, 9, and Grade 10 students in Reading, Writing, and Mathematics. For simplicity, this project will only review the Grade 6 data.

**Data Preparation**

The data preparation steps are outlined as follows:

1. Create a new dataframe using the 2023 data as a basis
2. Keep the following columns which are useful for prediction without adding too many features: 'School Language', 'Enrolment', Percentage of Students Whose First Language Is Not English, Percentage of Students Receiving Special Education Services, Percentage of Grade 6 Students Achieving the Provincial Standard in Reading, Percentage of Grade 6 Students Achieving the Provincial Standard in Writing, Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics, Change in Grade 6 Mathematics Achievement Over Three Years, Percentage of School-Aged Children Who Live in Low-Income Households, Percentage of Students Whose Parents Have No Degree, Diploma or Certificate
3. Remove rows with missing data
4. One Hot encode the categorical variables
5. Use Standard Scaler for the numerical variables







## Application and Code


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.decomposition import PCA

```


```python
#Load 2023 EQAO data
filepath_2023 = r"C:\Users\rossi\OneDrive\Data Science Projects\EQAO Test Scores\Data\raw\new_sif_data_table_2023_24prelim_en_january2026.xlsx - SIFprelim23-24_EN.csv"
df_2023 = pd.read_csv(filepath_2023)
```


```python
cols_to_keep = ['Board Name', 'School Language',
       'Enrolment',
       'Percentage of Students Whose First Language Is Not English',
       'Percentage of Students Whose First Language Is Not French',
       'Percentage of Students Who Are New to Canada from a Non-English Speaking Country',
       'Percentage of Students Who Are New to Canada from a Non-French Speaking Country',
       'Percentage of Students Receiving Special Education Services',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
       'Change in Grade 6 Reading Achievement Over Three Years',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
       'Change in Grade 6 Writing Achievement Over Three Years',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
       'Change in Grade 6 Mathematics Achievement Over Three Years',
       'Percentage of School-Aged Children Who Live in Low-Income Households',
       'Percentage of Students Whose Parents Have No Degree, Diploma or Certificate']

df = df_2023[cols_to_keep] #initialize raw dataframe with desired columns
```


```python
#Convert numeric rows which are currently strings to floats
cols_to_fix = ['Percentage of Students Whose First Language Is Not English',
       'Percentage of Students Whose First Language Is Not French',
       'Percentage of Students Who Are New to Canada from a Non-English Speaking Country',
       'Percentage of Students Who Are New to Canada from a Non-French Speaking Country',
       'Percentage of Students Receiving Special Education Services',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
       'Change in Grade 6 Reading Achievement Over Three Years',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
       'Change in Grade 6 Writing Achievement Over Three Years',
       'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
       'Change in Grade 6 Mathematics Achievement Over Three Years',
       'Percentage of School-Aged Children Who Live in Low-Income Households',
       'Percentage of Students Whose Parents Have No Degree, Diploma or Certificate']

for col in cols_to_fix:
  df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
```

```python
df[cols_to_fix] = df[cols_to_fix].astype(float)
```   


```python
df['Enrolment'] = pd.to_numeric(df['Enrolment'],errors = 'coerce')
```
   

**Exploratory Data Analysis**

Since we are preparing an unsupervised machine learning model, there is no need to split off portion of the dataset for training and testing purposes. All rows with missing data will be removed, for simplicity.

For EDA, we are going to create the following charts:

1. Pearson Correlation chart between selected socioeconomic factors and Grade outcomes in Reading, Writing, and Maths
2. Scatter Plot of Students in Low Income Housing vs Grade outcomes
3. Scatter Plot of Reading performance vs Maths performance
4. Scatter Plot of Percentage Change in Maths performance and the percentage of students meeting provincial standard in Maths


```python
# 1. Prepare the clean dataset (dropping rows with any missing values in relevant columns)
# List of columns needed for the plots and the model
relevant_cols = [
    'Percentage of School-Aged Children Who Live in Low-Income Households',
    'Percentage of Students Receiving Special Education Services',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
    'Change in Grade 6 Mathematics Achievement Over Three Years',
    'Percentage of Students Whose First Language Is Not English',
    'School Language',
    'Enrolment'
]

# Create df_clean for visualization and model
df_clean = df.dropna(subset=relevant_cols).copy()

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- PLOT 1: THE PEARSON CORRELATION HEATMAP ---
plt.figure(figsize=(10, 8))
corr = df_clean[relevant_cols].select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation: Demographics vs. Achievement Outcomes', fontsize=15)
plt.show()

# --- PLOT 2: Grade 6 Provincial Standard Distribution by Language and Income ---
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_clean,
    x='Percentage of School-Aged Children Who Live in Low-Income Households',
    y='Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
    hue='School Language',
    size='Enrolment',
    alpha=0.6,
    sizes=(20, 400)
)
plt.title('Demographic Island: Math Achievement vs. Low-Income Households', fontsize=15)
plt.xlabel('Low-Income Household (%)')
plt.ylabel('Grade 6 Math Achievement (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- PLOT 3: SOCIOECONOMIC INPUT BOXPLOTS (By Language) ---
plt.figure(figsize=(10, 6))
# Melding data for a side-by-side comparison
combined_df = df_clean.melt(id_vars='School Language',
                          value_vars=['Percentage of Students Receiving Special Education Services',
                                      'Percentage of School-Aged Children Who Live in Low-Income Households'],
                          var_name='Demographic', value_name='Percentage')

sns.boxplot(data=combined_df, x='Demographic', y='Percentage', hue='School Language')
plt.title('Distribution of School Needs by Language System', fontsize=15)
plt.xticks(rotation=15)
plt.show()

# --- PLOT 4: Distribution of 3-Year Grade 6 Achievement Change---
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Change in Grade 6 Mathematics Achievement Over Three Years'],
             kde=True, color='teal', bins=30)
plt.axvline(0, color='red', linestyle='--', label='No Change')
plt.title('Distribution of Math Achievement Trend (3-Year Change)', fontsize=15)
plt.xlabel('Percentage Point Change')
plt.legend()
plt.show()
```


![alt text](/img/posts/output_16_0.png "Pearson Correlation Heatmap")
    



    
![alt text](/img/posts/output_16_1.png "Grade 6 Provincial Standard Distribution by Language and Income")

    



    
![alt text](/img/posts/output_16_2.png "Socioeconomic Boxplots")

    



    
![alt text](/img/posts/output_16_3.png "Distribution of 3-Year Grade 6 Achievement Change")
    



```python
#K-Means Analysis

# 1. Define list of columns for consideration. Use a subset to reduce dimensionality
ml_columns = [
    'School Language', 'Enrolment',
    'Percentage of Students Whose First Language Is Not English',
    'Percentage of Students Receiving Special Education Services',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
    'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
    'Change in Grade 6 Mathematics Achievement Over Three Years',
    'Percentage of School-Aged Children Who Live in Low-Income Households',
    'Percentage of Students Whose Parents Have No Degree, Diploma or Certificate'
]

df_ml = df_clean[ml_columns].copy()

# 2. One-Hot Encode 'School Language'
df_encoded = pd.get_dummies(df_ml, columns=['School Language'], drop_first=True)

# 3. Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# 4. Silhouette Analysis Loop
k_range = range(2, 11)
silhouette_scores = []

print("Starting Silhouette Analysis...")
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)
    print(f"k={k} | Silhouette Score: {score:.4f}")

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', color='#5e2ced')
plt.title('Silhouette Analysis: Finding the Optimal Number of Clusters', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

    


    
![alt text](/img/posts/output_17_1.png "Silhouette Analysis")

    



```python
# Range of cluster sizes to visualize
range_n_clusters = [2, 3, 4, 5]

# Create a figure with subplots for each k
fig, ax_grid = plt.subplots(2, 2, figsize=(15, 12))
ax_grid = ax_grid.flatten()

for i, n_clusters in enumerate(range_n_clusters):
    ax = ax_grid[i]

    # 1. Initialize and fit the KMeans model
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_labels = clusterer.fit_predict(scaled_data)

    # 2. Calculate the average silhouette score and individual sample scores
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    sample_silhouette_values = silhouette_samples(scaled_data, cluster_labels)

    y_lower = 10
    for j in range(n_clusters):
        # Aggregate and sort silhouette scores for samples in cluster j
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
        ith_cluster_silhouette_values.sort()

        size_cluster_j = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_j

        # Assign a distinct color to this cluster
        color = cm.nipy_spectral(float(j) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the cluster number in the center of the silhouette
        ax.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

        # Update y_lower for the next cluster's plot
        y_lower = y_upper + 10

    # Plot formatting
    ax.set_title(f"Silhouette Analysis for k = {n_clusters}\nAvg Score: {silhouette_avg:.3f}", fontsize=12)
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")

    # Draw the vertical line for the average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Hide y-axis labels as they represent individual samples
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle("Silhouette Comparison for School Achievement & Demographic Clusters", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```


    
![alt text](/img/posts/output_18_0.png "Silhouette Analysis with SES Clusters")
    



```python
# 1. Run K-Means for k=3
kmeans_final = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(scaled_data)

# 2. Output the cluster column to your analysis dataframe
# We ensure df_ml is a proper copy to avoid SettingWithCopyWarning
df_ml['Cluster'] = cluster_labels

# 3. Create the Profiler Table
# Transposing the means makes it much easier to compare the 3 Islands side-by-side
cluster_profile = df_ml.groupby('Cluster').mean(numeric_only=True).T

# Add total school counts to the bottom of the profile
counts = df_ml['Cluster'].value_counts().sort_index()
cluster_profile.loc['Total_Schools_in_Cluster'] = counts.values

print("--- Cluster Feature Profiles (Averages) ---")
print(cluster_profile)

# 5. Language Breakdown per Cluster
lang_pct = pd.crosstab(df_ml['Cluster'], df_ml['School Language'], normalize='index') * 100
print("\n--- Language Distribution (%) per Cluster ---")
print(lang_pct)
```

    --- Cluster Feature Profiles (Averages) ---
    Cluster                                                       0            1  \
    Enrolment                                            424.979307   333.839373   
    Percentage of Students Whose First Language Is ...    17.620797    18.107738   
    Percentage of Students Receiving Special Educat...    12.215210    16.980411   
    Percentage of Grade 6 Students Achieving the Pr...    87.842214    69.056807   
    Percentage of Grade 6 Students Achieving the Pr...    86.453699    66.036239   
    Percentage of Grade 6 Students Achieving the Pr...    58.053802    29.517140   
    Change in Grade 6 Mathematics Achievement Over ...     6.797724    -3.004897   
    Percentage of School-Aged Children Who Live in ...     7.078117    12.895201   
    Percentage of Students Whose Parents Have No De...     2.920848     7.813908   
    Total_Schools_in_Cluster                            1933.000000  1021.000000   
    
    Cluster                                                      2  
    Enrolment                                           270.200803  
    Percentage of Students Whose First Language Is ...   54.313253  
    Percentage of Students Receiving Special Educat...   11.811245  
    Percentage of Grade 6 Students Achieving the Pr...   97.417671  
    Percentage of Grade 6 Students Achieving the Pr...   77.269076  
    Percentage of Grade 6 Students Achieving the Pr...   56.718876  
    Change in Grade 6 Mathematics Achievement Over ...    7.787149  
    Percentage of School-Aged Children Who Live in ...    8.212851  
    Percentage of Students Whose Parents Have No De...    3.096386  
    Total_Schools_in_Cluster                            249.000000  
    
    --- Language Distribution (%) per Cluster ---
    School Language  English  French
    Cluster                         
    0                  100.0     0.0
    1                  100.0     0.0
    2                    0.0   100.0
    


```python
# Plot Grade 6 Maths Performance vs Percentage of Students in Low Income Households to show stratification of clusters
sns.set_theme(style="whitegrid", palette="viridis")
plt.figure(figsize=(14, 8))

plot = sns.scatterplot(
    data=df_ml,
    x='Percentage of School-Aged Children Who Live in Low-Income Households',
    y='Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
    hue='Cluster',
    size='Enrolment',
    sizes=(50, 600), # Makes bubbles prominent
    alpha=0.7,
    palette='viridis',
    edgecolor='w'
)

# Add a horizontal line at 50% to represent the Province's baseline for success
plt.axhline(df_ml['Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics'].mean(),
            color='red', linestyle='--', alpha=0.5, label='Provincial Math Mean')

plt.title('Ontario School Clusters: The Intersection of Socioeconomics and Achievement', fontsize=18, pad=20)
plt.xlabel('Socioeconomic Challenge (% Low-Income Households)', fontsize=13)
plt.ylabel('Academic Performance (% Meeting Math Standard)', fontsize=13)

plt.legend(title='Identified Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('cluster_summary.png', dpi=300) # Save for your README
plt.show()
```


    
![alt text](/img/posts/output_20_0.png "Summary of Identified Clusters")
    



```python
# Use Dimensionality Reduction (PCA) to display clusters on a 2D plot

pca = PCA(n_components=2) #2 components for 2D plot
X_pca = pca.fit_transform(scaled_data) #scaled_data from earlier

# Transform the Centroids like features
centroids_pca = pca.transform(kmeans_final.cluster_centers_)

plt.figure(figsize=(12, 8))

# Scatter plot of the schools colored by existing cluster_labels
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=cluster_labels,
    palette='viridis',
    alpha=0.6,
    s=60,
    edgecolor='w',
    linewidth=0.5
)

# Plotting the Centroids (The "X" markers)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    color='red',
    marker='X',
    s=250,
    label='Centroids',
    edgecolor='black',
    zorder=10 # Ensures centroids sit on top of the data points
)

# Formatting
plt.title('2D Cluster Projection: EQAO K-Means Analysis', fontsize=16, pad=20)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)', fontsize=12)

plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('PCA_summary.png', dpi=300)
plt.show()

# Verification for your documentation
total_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"The 2D PCA plot captures {total_var:.2f}% of the total multi-dimensional variance.")
```


    
![alt text](/img/posts/output_21_0.png "PCA Analysis Summary")
    


    The 2D PCA plot captures 47.62% of the total multi-dimensional variance.
    

## Results Analysis

In order to select the optimal number of clusters for the K-Means analysis, I
calculated the average Silhouette score across a range of assumed number of clusters. The Silhouette score is a method to compare cluster quality betewen models. The peak score was found for a model with 3 clusters, thus a K means model with 3 clusters was chosen.

A silhouette analysis was then performed for K-Means models by varying the assumged number of clusters from 2-5. All the models showed clusters with silhouette scores primarily in the positive range, with one cluster showing negative values. For the K Means model with 3 clusters, clusters 0, 1, and 2 all exceeded the average Silhouette value of 0.24, with peaks exceeding 0.4.

Cluster 0 was the found to be the largest, followed by Cluster 1 and 2. Cluster 1's silhouette analysis has a negative value, signifying that there are schools that straddle two clusters but have been assigned to Cluster 1.

The scatter plot of Math performance vs % of students in low-income houses provides some insight into the cluster makeup. Cluster 0, the largest, is comprised primarily of English speaking schools whose Maths performances are average to above average (mean > 58.1%). Cluster 1 comprises entirely English speaking schools but with a worse average performance in Maths (29.5%). Cluster 2 comprises only French-speaking schools.

The scatter plot also shows a trend between the percentage of students passing the Maths exam and the percentage of students living in low-income houses. The data trends down and to the right, suggesting that as the percentage of students living in low-income houses increases the percentage of students reaching provindial standard decreases.

The PCA analysis shows 3 clusters, each over lapping with the other. This suggests that these clusters aren't entirely distinct but rather there are a number of schools that straddle two clusters. 47% of the total-dimensional variance was explained by the first two components of the PCA.



## Next Steps

In a future extension to this project, the following steps could be taken to perform a deeper dive into the data:

*   In addition to K-Means, other clustering methods could also be used, e.g. DBSCAN or t-SNE
* Incorporate school funding data or teacher-to-student ratios as potentially strong, new features
*   Run the clustering analysis on other school years to see if similar clusters are formed



