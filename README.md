# 🛒 ShopSense

## E-Commerce Customer Segmentation Using Machine Learning

> Turning Customer Data into Meaningful Customer Insights

------------------------------------------------------------------------

## 📌 Project Overview

**ShopSense** is an e-commerce customer segmentation project that uses
**unsupervised machine learning** to identify meaningful customer groups
based on demographic characteristics, purchasing behaviour, digital
engagement, and campaign response.

The project uses a customer dataset containing **2,240 records and 22
original attributes**. The analysis includes data preprocessing, feature
engineering, outlier treatment, feature scaling, PCA, **K-Means
Clustering**, and **Agglomerative Hierarchical Clustering**.

The final analysis identifies **four customer segments: C0, C1, C2 and
C3**. The final cluster characterisation is based on the **Agglomerative
Clustering labels**.

------------------------------------------------------------------------

## 🎯 Problem Statement

E-commerce customers do not all behave in the same way. Applying a
common marketing strategy across the entire customer base can overlook
differences in purchasing behaviour, customer value, digital engagement,
purchase channels, household characteristics, and campaign response.

The objective of ShopSense is to use **unsupervised machine learning**
to discover hidden patterns in customer data and create meaningful
customer segments that can support targeted marketing, customer
engagement, and retention strategies.

------------------------------------------------------------------------

## 🎯 Objectives

1.  Explore and understand the customer dataset.
2.  Handle missing values and prepare the data for machine learning.
3.  Engineer meaningful customer-level features.
4.  Encode categorical variables.
5.  Detect and remove selected outliers.
6.  Standardise the clustering features.
7.  Apply PCA for dimensionality reduction.
8.  Determine an appropriate number of clusters.
9.  Apply **K-Means Clustering**.
10. Apply **Agglomerative Hierarchical Clustering**.
11. Analyse and interpret the resulting customer segments.
12. Translate cluster characteristics into business insights.

------------------------------------------------------------------------

# 📊 Dataset

The original dataset contains **2,240 customers and 22 attributes**.

### Feature Groups

**Demographics** - `ID` - `Year_Birth` - `Education` -
`Marital_Status` - `Income` - `Kidhome` - `Teenhome` - `Dt_Customer`

**Product Spending** - `MntWines` - `MntFruits` - `MntMeatProducts` -
`MntFishProducts` - `MntSweetProducts` - `MntGoldProds`

**Purchase Frequency / Digital Activity** - `NumDealsPurchases` -
`NumWebPurchases` - `NumCatalogPurchases` - `NumStorePurchases` -
`NumWebVisitsMonth`

**Customer Activity / Response** - `Recency` - `Complain` - `Response`

------------------------------------------------------------------------

# 🧹 Data Preprocessing

### Missing Values

The dataset contained **24 missing values in `Income`**. Median
imputation was applied.

``` python
data["Income"] = data["Income"].fillna(data["Income"].median())
```

### Feature Engineering

The notebook created:

-   **Age**
-   **Customer_Tenure_Days**
-   **Total_Spending**
-   **Total_Children**

``` python
data["Age"] = 2026 - data["Year_Birth"]

data["Total_Spending"] = (
    data["MntWines"] +
    data["MntFruits"] +
    data["MntMeatProducts"] +
    data["MntFishProducts"] +
    data["MntSweetProducts"] +
    data["MntGoldProds"]
)

data["Total_Children"] = (
    data["Kidhome"] +
    data["Teenhome"]
)
```

`Dt_Customer` was converted to a date and used to calculate customer
tenure.

### Categorical Transformation

Education was consolidated into:

-   Graduate
-   Postgraduate
-   Undergraduate

Marital status was transformed into `Living_With`:

-   Partner
-   Single

One-hot encoding was applied to:

``` python
cat_cols = ["Education", "Living_With"]
```

### Feature Selection

The notebook removed the original variables that were redundant after
feature engineering/transformation, including:

``` text
ID
Year_Birth
Marital_Status
Kidhome
Teenhome
MntMeatProducts
Dt_Customer
MntWines
MntFruits
MntFishProducts
MntSweetProducts
MntGoldProds
```

After one-hot encoding, the clustering dataset contained **18
features**.

------------------------------------------------------------------------

# ⚠️ Outlier Treatment

Potential outliers were examined using:

-   Income
-   Recency
-   Response
-   Age
-   Total Spending
-   Total Children

The notebook retained:

``` python
data_cleaned = data_cleaned[data_cleaned["Age"] < 90]
data_cleaned = data_cleaned[data_cleaned["Income"] < 600000]
```

  Stage                        Records
  ------------------------ -----------
  Before outlier removal         2,240
  After outlier removal      **2,236**
  Removed                        **4**

------------------------------------------------------------------------

# ⚖️ Feature Scaling

The prepared features were standardised using `StandardScaler`.

``` python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The scaled matrix contained **2,236 observations × 18 features**.

------------------------------------------------------------------------

# 📉 Principal Component Analysis (PCA)

PCA was used to reduce the dimensionality of the scaled feature space.

  Component                 Explained Variance
  ----------------------- --------------------
  PCA 1                            **23.163%**
  PCA 2                            **11.385%**
  PCA 3                            **10.406%**
  PCA 1 + PCA 2                    **34.548%**
  PCA 1 + PCA 2 + PCA 3            **45.154%**

The first three principal components were used as the clustering
representation.

------------------------------------------------------------------------

# 🔍 Selecting the Number of Clusters

The notebook evaluated **K = 1 to 10** using the **Elbow Method /
WCSS**.

`KneeLocator` identified:

``` text
Optimal K = 4
```

Therefore, **four clusters** were selected for the final analysis.

Silhouette scores were also calculated for **K = 2 to K = 10** as an
additional clustering diagnostic. The selection of K=4 is based on the
**Elbow/Knee analysis**, not a claim that K=4 maximised the silhouette
score.

------------------------------------------------------------------------

# 🤖 Machine Learning Models

## K-Means Clustering

K-Means was implemented using:

``` python
KMeans(
    n_clusters=4,
    random_state=42
)
```

The model partitions customers into four groups based on their
similarity in the PCA feature space.

## Agglomerative Hierarchical Clustering

Agglomerative clustering was implemented using:

``` python
AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)
```

Configuration:

-   **Clusters:** 4
-   **Linkage:** Ward

The final cluster-level customer characterisation in the notebook uses
the **Agglomerative Clustering labels**.

------------------------------------------------------------------------

# 📊 Final Customer Segmentation

> **C0, C1, C2 and C3 are cluster identifiers, not rankings.**

## 🔴 C0 --- Family-Oriented Lower-Spend Customers

  Metric                         Mean
  --------------------- -------------
  Income                    39,680.58
  Recency                       48.91
  Deal Purchases                 2.59
  Web Purchases                  3.15
  Catalogue Purchases            0.97
  Store Purchases                4.14
  Web Visits / Month             6.31
  Response                      7.62%
  Age                           55.67
  Total Spending               221.96
  Total Children                 1.24
  Customer Tenure         342.94 days

C0 combines relatively lower income and spending with higher average
children and relatively high website activity. Campaign response is
comparatively low.

**Suggested focus:** family-oriented offers, personalised
recommendations, retargeting, cart recovery, and conversion-focused
promotions.

------------------------------------------------------------------------

## 🟠 C1 --- High-Spending Multi-Channel Customers

  Metric                         Mean
  --------------------- -------------
  Income                    72,808.45
  Recency                       49.20
  Deal Purchases                 1.96
  Web Purchases                  5.69
  Catalogue Purchases            5.50
  Store Purchases                8.66
  Web Visits / Month             3.58
  Response                     16.67%
  Age                           59.49
  Total Spending             1,236.59
  Total Children                 0.51
  Customer Tenure         369.72 days

C1 has the highest average income and total spending and shows strong
purchasing activity across web, catalogue, and store channels.

**Suggested focus:** loyalty rewards, repeat-purchase incentives,
cross-selling, personalised recommendations, and omnichannel engagement.

------------------------------------------------------------------------

## 🔵 C2 --- Digital-Engaged Lower-Spend Customers

  Metric                         Mean
  --------------------- -------------
  Income                    36,960.14
  Recency                       48.32
  Deal Purchases                 2.59
  Web Purchases                  2.71
  Catalogue Purchases            0.84
  Store Purchases                3.62
  Web Visits / Month             6.66
  Response                     14.19%
  Age                           55.69
  Total Spending               165.70
  Total Children                 1.27
  Customer Tenure         338.78 days
  Living With Single           99.32%

C2 has the lowest average income and total spending but the highest
website-visit frequency at **6.66 visits per month**.

**Suggested focus:** digital-first campaigns, personalised web
recommendations, digital coupons, retargeting, cart-abandonment
campaigns, and online conversion offers.

------------------------------------------------------------------------

## 🟢 C3 --- High-Value Highly Responsive Customers

  Metric                         Mean
  --------------------- -------------
  Income                    70,722.68
  Recency                       50.50
  Deal Purchases                 1.86
  Web Purchases                  5.79
  Catalogue Purchases            5.01
  Store Purchases                8.43
  Web Visits / Month             3.73
  Response                     32.01%
  Age                           58.93
  Total Spending             1,190.39
  Total Children                 0.46
  Customer Tenure         376.28 days
  Living With Single             100%

C3 combines high income and spending with strong multi-channel
purchasing activity. It also has the highest campaign response rate at
**32.01%**.

**Suggested focus:** premium loyalty, exclusive offers, personalised
service, high-value cross-selling, and retention-focused campaigns.

------------------------------------------------------------------------

# 📋 Cluster Comparison

  ------------------------------------------------------------------------------------
  Characteristic   C0                C1              C2                 C3
  ---------------- ----------------- --------------- ------------------ --------------
  Income           Lower             Highest         Lowest             High

  Total Spending   Lower             Highest         Lowest             High

  Website Visits   High              Low             Highest            Low

  Web Purchases    Moderate          High            Low                Highest

  Store Purchases  Moderate          Highest         Lower              High

  Campaign         Lowest            Moderate        Moderate           Highest
  Response                                                              

  Total Children   Higher            Low             Highest            Low

  Main Profile     Family-oriented   High-spending   Digital-engaged,   High-value,
                                     multi-channel   lower-spend        highly
                                                                        responsive
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 💡 Key Insights

### Customer Value

Average total spending varies substantially:

``` text
C2 → 165.70
C0 → 221.96
C3 → 1190.39
C1 → 1236.59
```

### Digital Engagement

Average website visits per month:

``` text
C0 → 6.31
C1 → 3.58
C2 → 6.66
C3 → 3.73
```

C2 has the highest website-visit activity but the lowest average total
spending, showing that browsing activity does not necessarily translate
into high spending.

### Campaign Response

``` text
C0 → 7.62%
C1 → 16.67%
C2 → 14.19%
C3 → 32.01%
```

C3 has the highest campaign response, while C0 has the lowest.

### Purchase Channels

C1 and C3 show substantially higher web, catalogue, and store purchase
frequencies than C0 and C2.

### Household Composition

C0 and C2 have higher average `Total_Children`, while C1 and C3 have
lower values.

------------------------------------------------------------------------

# 📣 Segment-Specific Business Strategy

  -----------------------------------------------------------------------
  Segment                 Main Focus              Suggested Actions
  ----------------------- ----------------------- -----------------------
  **C0**                  Improve conversion      Family bundles,
                                                  personalised offers,
                                                  retargeting, cart
                                                  recovery

  **C1**                  Loyalty and repeat      Loyalty rewards, repeat
                          value                   purchases,
                                                  cross-selling,
                                                  omnichannel engagement

  **C2**                  Digital conversion      Digital coupons, online
                                                  offers,
                                                  recommendations,
                                                  retargeting

  **C3**                  Retention and value     Premium loyalty,
                                                  exclusive offers,
                                                  personalised service,
                                                  cross-selling
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🔄 End-to-End ML Pipeline

``` text
Customer Dataset
       │
       ▼
Data Exploration
       │
       ▼
Missing Value Treatment
       │
       ▼
Feature Engineering
       │
       ├── Age
       ├── Customer Tenure
       ├── Total Spending
       └── Total Children
       │
       ▼
Categorical Transformation
       │
       ▼
One-Hot Encoding
       │
       ▼
Feature Selection
       │
       ▼
Outlier Treatment
       │
       ▼
StandardScaler
       │
       ▼
PCA
18 Features → 3 Principal Components
       │
       ├──────────────────────┐
       ▼                      ▼
   K-Means              Agglomerative
   K = 4                K = 4, Ward
       │                      │
       └──────────┬───────────┘
                  ▼
            Cluster Analysis
                  │
                  ▼
             C0 | C1 | C2 | C3
                  │
                  ▼
          Business Insights
```

------------------------------------------------------------------------

# 📊 Visualisations

The notebook includes:

-   Pair plots
-   Correlation heatmap
-   2D PCA visualisation
-   3D PCA visualisation
-   Elbow/WCSS curve
-   Silhouette-score analysis
-   Combined WCSS and silhouette analysis
-   3D K-Means cluster visualisation
-   3D Agglomerative cluster visualisation
-   Cluster-count visualisation
-   Income vs. Total Spending scatter plot
-   Cluster-level summary analysis

------------------------------------------------------------------------

# 🧪 Model Summary

  Component                        Implementation
  -------------------------------- -------------------------------
  Learning Type                    Unsupervised Learning
  Task                             Customer Segmentation
  K-Means                          4 clusters, `random_state=42`
  Agglomerative                    4 clusters, Ward linkage
  Cluster Selection                Elbow/Knee analysis
  Silhouette Analysis              K = 2 to 10
  Scaling                          StandardScaler
  Dimensionality Reduction         PCA
  PCA Components for Clustering    3
  Final Cluster Characterisation   Agglomerative labels
  Final Segments                   C0, C1, C2, C3

------------------------------------------------------------------------

# 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   PCA
-   StandardScaler
-   K-Means Clustering
-   Agglomerative Hierarchical Clustering
-   KneeLocator / Kneed
-   Jupyter Notebook

------------------------------------------------------------------------

# 📁 Project Structure

``` text
ShopSense/
│
├── data/
│   └── smartcart_customers.csv
│
├── notebook/
│   └── ShopSense_Clustering.ipynb
│
├── report/
│   └── ShopSense_Corrected_Customer_Segmentation_Report.pdf
│
└── README.md
```

------------------------------------------------------------------------

# ⚠️ Interpretation Notes

-   **C0, C1, C2 and C3 are cluster identifiers, not rankings.**
-   The segment names are business interpretations of the cluster-level
    characteristics.
-   **K = 4** was selected using the Elbow/Knee analysis.
-   Silhouette scores were used as an additional diagnostic, not as the
    stated selection criterion for K=4.
-   Both **K-Means** and **Agglomerative Hierarchical Clustering** were
    implemented.
-   The final cluster-level characterisation uses the **Agglomerative
    Clustering labels**.
-   The separate income-versus-spending segmentation is intentionally
    excluded from the final segmentation presentation.

------------------------------------------------------------------------

# 🚀 Project Outcome

ShopSense transforms customer-level e-commerce data into four
interpretable behavioural profiles:

  Cluster   Profile
  --------- ----------------------------------------
  **C0**    Family-Oriented Lower-Spend Customers
  **C1**    High-Spending Multi-Channel Customers
  **C2**    Digital-Engaged Lower-Spend Customers
  **C3**    High-Value Highly Responsive Customers

These segments provide a data-driven foundation for personalised
marketing, digital conversion, loyalty initiatives, customer retention,
and targeted customer engagement.

------------------------------------------------------------------------

# ✅ Conclusion

The **ShopSense -- E-Commerce Customer Segmentation Using Machine
Learning** project demonstrates a complete unsupervised-learning
workflow covering data preprocessing, feature engineering, outlier
treatment, standardisation, PCA, cluster selection, K-Means clustering,
Agglomerative clustering, and business interpretation.

Using **K-Means and Agglomerative Hierarchical Clustering**, with **K=4
selected through the Elbow/Knee analysis**, the project identifies four
distinct customer profiles.

The results highlight differences in customer spending, income, website
engagement, purchase channels, campaign response, household composition,
and tenure. These differences can support more targeted and data-driven
customer strategies.

> **ShopSense --- Turning Customer Data into Meaningful Customer
> Insights.**
