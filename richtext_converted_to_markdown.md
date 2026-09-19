Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML``   # 🛒 ShopSense  ## E-Commerce Customer Segmentation Using Machine Learning  > **Turning Customer Data into Meaningful Customer Insights**  ---  ## 📌 1. Project Overview  **ShopSense** is an e-commerce customer segmentation project that uses **unsupervised machine learning** to identify meaningful groups of customers based on their demographics, purchasing behaviour, digital engagement, and campaign response.  The project uses a customer dataset containing **2,240 records and 22 original attributes**. The objective is to discover hidden patterns in customer behaviour and translate the resulting clusters into actionable business insights for personalised marketing and customer retention.  The analysis applies **both K-Means and Agglomerative Hierarchical Clustering** and uses dimensionality reduction and cluster-evaluation techniques to support the segmentation process.  ---  ## 🎯 2. Problem Statement  ShopSense currently applies generic marketing and engagement strategies across its customer base without sufficiently understanding the differences between customer groups.  This can lead to:  - Inefficient marketing efforts  - Missed opportunities to retain high-value customers  - Limited personalisation  - Difficulty identifying different purchasing patterns  - Delayed identification of customers requiring targeted engagement  The goal is to develop an **intelligent customer segmentation system using unsupervised machine learning** that groups customers according to similarities in their purchasing behaviour, engagement levels, demographics, and loyalty-related indicators.  ---  ## 🎯 3. Project Objectives  The project aims to:  1. Explore and understand the customer dataset.  2. Handle missing values and prepare the data for analysis.  3. Engineer meaningful customer-level features.  4. Encode categorical variables and scale the data.  5. Reduce dimensionality using Principal Component Analysis (PCA).  6. Determine an appropriate number of clusters.  7. Apply **K-Means Clustering**.  8. Apply **Agglomerative Hierarchical Clustering**.  9. Characterise the resulting customer segments.  10. Derive actionable business insights from the clusters.  ---  # 📊 4. Dataset Overview  The original dataset contains:  | Property | Value |  |---|---:|  | Customer Records | **2,240** |  | Original Attributes | **22** |  | Numerical Attributes | **18** |  | Categorical Attributes | **3** |  | Missing Values | **24 in Income** |  | Dataset | `smartcart_customers.csv` |  Each row represents one customer and contains information about demographics, product spending, purchase frequency, website activity, recency, complaints, and campaign response.  ### Original Feature Groups  #### 👤 Customer Demographics  - `ID`  - `Year_Birth`  - `Education`  - `Marital_Status`  - `Income`  - `Kidhome`  - `Teenhome`  - `Dt_Customer`  #### 🛍️ Purchase Behaviour — Amount Spent  - `MntWines`  - `MntFruits`  - `MntMeatProducts`  - `MntFishProducts`  - `MntSweetProducts`  - `MntGoldProds`  #### 🛒 Purchase Behaviour — Frequency  - `NumDealsPurchases`  - `NumWebPurchases`  - `NumCatalogPurchases`  - `NumStorePurchases`  - `NumWebVisitsMonth`  #### 📣 Customer Feedback & Response  - `Recency`  - `Complain`  - `Response`  ---  # 🧹 5. Data Preprocessing  ## 5.1 Missing-Value Treatment  The dataset contained **24 missing values in `Income`**.  These missing values were handled using **median imputation**:  ```python  data["Income"] = data["Income"].fillna(data["Income"].median())   ``

After imputation, no missing values remained in the dataset.

5.2 Feature Engineering
-----------------------

Several new features were created to better represent customer behaviour.

### Age

Customer age was derived from the year of birth:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data["Age"] = 2026 - data["Year_Birth"]   `

### Customer Tenure

The customer enrolment date was converted to a date format, and tenure was calculated relative to the latest customer enrolment date in the dataset:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data["Dt_Customer"] = pd.to_datetime(      data["Dt_Customer"],      dayfirst=True  )  reference_date = data["Dt_Customer"].max()  data["Customer_Tenure_Days"] = (      reference_date - data["Dt_Customer"]  ).dt.days   `

### Total Spending

Spending across the six product categories was aggregated into:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data["Total_Spending"] = (      data["MntWines"] +      data["MntFruits"] +      data["MntMeatProducts"] +      data["MntFishProducts"] +      data["MntSweetProducts"] +      data["MntGoldProds"]  )   `

### Total Children

The number of children and teenagers was combined:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data["Total_Children"] = (      data["Kidhome"] +      data["Teenhome"]  )   `

5.3 Categorical Feature Transformation
--------------------------------------

Education categories were consolidated into three groups:

*   Graduate
    
*   Postgraduate
    
*   Undergraduate
    

Marital-status information was transformed into a simplified Living\_With feature:

*   Partner
    
*   Single
    

The categorical variables used for encoding were:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   cat_cols = ["Education", "Living_With"]   `

One-hot encoding was then applied.

5.4 Feature Selection
---------------------

The following columns were removed before clustering because their information was either redundant with engineered features, unsuitable as direct clustering variables, or represented the original form of transformed features:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ID  Year_Birth  Marital_Status  Kidhome  Teenhome  MntMeatProducts  Dt_Customer  MntWines  MntFruits  MntFishProducts  MntSweetProducts  MntGoldProds   `

After feature preparation, the clustering dataset initially contained **15 features**, which became **18 features after one-hot encoding**.

5.5 Outlier Treatment
---------------------

Potential outliers were examined using selected variables:

*   Income
    
*   Recency
    
*   Response
    
*   Age
    
*   Total\_Spending
    
*   Total\_Children
    

Two filtering conditions were applied:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data_cleaned = data_cleaned[      data_cleaned["Age"] < 90  ]  data_cleaned = data_cleaned[      data_cleaned["Income"] < 600000  ]   `

The dataset size changed from:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Before outlier removal : 2240  After outlier removal  : 2236   `

Thus, **4 observations were removed as outliers**.

⚖️ 6. Feature Scaling
=====================

Because clustering algorithms are distance-based, the prepared features were standardised using StandardScaler.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   scaler = StandardScaler()  X_scaled = scaler.fit_transform(X)   `

The resulting scaled feature matrix contained:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   2236 observations × 18 features   `

This ensured that variables with larger numerical ranges did not disproportionately influence the clustering process.

📉 7. Dimensionality Reduction Using PCA
========================================

Principal Component Analysis (**PCA**) was used to reduce the dimensionality of the scaled feature space and provide a lower-dimensional representation for visualisation and clustering.

### 2D PCA

The first two principal components explained:

ComponentExplained VariancePCA 1**23.16%**PCA 2**11.39%**

Combined, the first two components explained approximately **34.55%** of the variance.

### 3D PCA

Three principal components were then used:

ComponentExplained VariancePCA 1**23.16%**PCA 2**11.39%**PCA 3**10.41%Total45.16%**

The three-dimensional PCA representation was subsequently used for clustering and visualisation.

🔍 8. Determining the Number of Clusters
========================================

The notebook evaluated different values of **K from 1 to 10** using the **Elbow Method**.

### Elbow Method

Within-Cluster Sum of Squares (**WCSS**) was calculated for each value of K.

The KneeLocator method identified:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Optimal K = 4   `

Therefore, **four clusters** were used for the final customer segmentation.

Silhouette Score
----------------

Silhouette scores were also calculated for **K = 2 through K = 10** to provide an additional measure of cluster structure.

The notebook uses the silhouette analysis as a supplementary diagnostic alongside the elbow method. Importantly, the silhouette curve does not show its maximum at K=4; therefore, K=4 should be reported as the solution selected from the **Elbow/Knee analysis**, rather than claiming that it produced the highest silhouette score.

🤖 9. Machine Learning Models
=============================

Two unsupervised clustering algorithms were applied.

9.1 K-Means Clustering
----------------------

K-Means clustering was implemented with:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   KMeans(      n_clusters=4,      random_state=42  )   `

The model generated four clusters using the three-dimensional PCA representation.

K-Means provides a centroid-based partition of customers and is useful for identifying groups with similar overall behavioural profiles.

9.2 Agglomerative Hierarchical Clustering
-----------------------------------------

Agglomerative Hierarchical Clustering was also implemented:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   AgglomerativeClustering(      n_clusters=4,      linkage="ward"  )   `

The model used:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Number of clusters = 4  Linkage            = Ward   `

Agglomerative clustering starts with individual observations and progressively merges similar observations/clusters, producing a hierarchical grouping structure.

The resulting cluster labels from Agglomerative Clustering were used for the final cluster characterisation in the notebook.

📊 10. Final Customer Segmentation
==================================

The final cluster characterisation is based on the **Agglomerative Clustering labels** and the mean values calculated for each cluster.

The four clusters are labelled **C0, C1, C2 and C3**.

> Cluster numbers are identifiers generated by the algorithm and do not represent a ranking.

🔴 C0 — Family-Oriented Lower-Spend Customers
---------------------------------------------

### Key characteristics

MetricCluster C0 MeanIncome**39,680.58**Recency**48.91 days**Deal Purchases**2.59**Web Purchases**3.15**Catalogue Purchases**0.97**Store Purchases**4.14**Web Visits / Month**6.31**Response Rate**7.62%**Age**55.67**Total Spending**221.96**Total Children**1.24**Customer Tenure**342.94 days**

### Interpretation

C0 customers have comparatively **lower income and lower total spending**, while having a relatively high number of children and frequent website visits. Their web and store purchasing activity is moderate, while campaign response is relatively low.

### Business Insight

This segment combines **family-oriented characteristics with relatively high browsing activity but lower spending and campaign response**.

### Suggested Strategy

*   Family-oriented product bundles
    
*   Personalised recommendations
    
*   Website retargeting
    
*   Cart-recovery campaigns
    
*   Offers aimed at improving conversion
    

🟠 C1 — High-Spending Multi-Channel Customers
=============================================

### Key characteristics

MetricCluster C1 MeanIncome**72,808.45**Recency**49.20 days**Deal Purchases**1.96**Web Purchases**5.69**Catalogue Purchases**5.50**Store Purchases**8.66**Web Visits / Month**3.58**Response Rate**16.67%**Age**59.49**Total Spending**1,236.59**Total Children**0.51**Customer Tenure**369.72 days**

### Interpretation

C1 has substantially higher **income and total spending** than C0 and C2. Customers in this group purchase through multiple channels, with particularly high store, web, and catalogue purchasing activity.

### Business Insight

This segment represents **high-spending, established customers with strong multi-channel purchasing behaviour**.

### Suggested Strategy

*   Loyalty rewards
    
*   Repeat-purchase incentives
    
*   Cross-selling
    
*   Personalised product recommendations
    
*   Omnichannel engagement
    

🔵 C2 — Digital-Engaged Lower-Spend Customers
=============================================

### Key characteristics

MetricCluster C2 MeanIncome**36,960.14**Recency**48.32 days**Deal Purchases**2.59**Web Purchases**2.71**Catalogue Purchases**0.84**Store Purchases**3.62**Web Visits / Month**6.66**Response Rate**14.19%**Age**55.69**Total Spending**165.70**Total Children**1.27**Customer Tenure**338.78 days**Living With Single**99.32%**

### Interpretation

C2 has the **lowest average income and total spending** among the four clusters while recording the **highest average number of website visits per month**.

The cluster is also overwhelmingly represented by customers classified as Living\_With = Single.

### Business Insight

C2 represents a group with **high website engagement but relatively low purchasing and spending levels**.

### Suggested Strategy

*   Digital-first campaigns
    
*   Personalised web recommendations
    
*   Digital coupons
    
*   Retargeting
    
*   Cart-abandonment campaigns
    
*   Conversion-focused online promotions
    

🟢 C3 — High-Value Highly Responsive Customers
==============================================

### Key characteristics

MetricCluster C3 MeanIncome**70,722.68**Recency**50.50 days**Deal Purchases**1.86**Web Purchases**5.79**Catalogue Purchases**5.01**Store Purchases**8.43**Web Visits / Month**3.73**Response Rate**32.01%**Age**58.93**Total Spending**1,190.39**Total Children**0.46**Customer Tenure**376.28 days**Living With Single**100%**

### Interpretation

C3 combines **high income, high total spending, strong multi-channel purchasing behaviour, and the highest campaign response rate** among all four clusters.

The average campaign response is approximately **32.01%**, compared with 16.67% for C1, 14.19% for C2, and 7.62% for C0.

### Business Insight

C3 represents the clearest **high-value and highly campaign-responsive customer segment** in the clustering results.

### Suggested Strategy

*   Premium loyalty programmes
    
*   Exclusive offers
    
*   Personalised service
    
*   High-value cross-selling
    
*   Retention-focused campaigns
    

📋 11. Cluster Comparison
=========================

CharacteristicC0C1C2C3IncomeLower**HighestLowest**HighTotal SpendingLower**HighestLowest**HighWeb VisitsHighLow**Highest**LowWeb PurchasesModerateHighLow**Highest**Store PurchasesModerate**Highest**LowerHighCampaign Response**Lowest**ModerateModerate**Highest**Total ChildrenHigherLow**Highest**LowCustomer ProfileFamily-orientedHigh-spending multi-channelDigital-engaged, lower-spendHigh-value, highly responsive

📈 12. Important Quantitative Findings
======================================

### Spending

The average total spending varies considerably across the clusters:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   C2  → 165.70  C0  → 221.96  C3  → 1190.39  C1  → 1236.59   `

This shows a clear separation between the lower-spending groups **C0/C2** and the substantially higher-spending groups **C1/C3**.

### Income

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   C2  → 36,960.14  C0  → 39,680.58  C3  → 70,722.68  C1  → 72,808.45   `

C1 and C3 therefore have considerably higher average income than C0 and C2.

### Digital Engagement

Average monthly website visits are:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   C0 → 6.31  C1 → 3.58  C2 → 6.66  C3 → 3.73   `

C2 has the highest average website-visit activity, followed by C0.

### Campaign Response

Average response rates are:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   C0 → 7.62%  C1 → 16.67%  C2 → 14.19%  C3 → 32.01%   `

C3 has the strongest campaign response, while C0 has the weakest.

💡 13. Key Business Insights
============================

### 1\. Customer value differs substantially across segments

C1 and C3 have average total spending above **1,190**, whereas C0 and C2 average below **222**.

This indicates a substantial difference in customer purchasing value across the identified clusters.

### 2\. High spending is associated with broader purchasing activity

C1 and C3 show substantially higher web, catalogue, and store purchase frequencies than C0 and C2.

### 3\. Website engagement does not necessarily equal high spending

C2 records the highest website-visit frequency but the **lowest total spending**.

This indicates that website engagement alone does not necessarily translate into high customer value.

### 4\. Campaign responsiveness varies considerably

C3 has a response rate of approximately **32%**, more than four times the response rate of C0.

This makes campaign responsiveness an important differentiating characteristic between the segments.

### 5\. Family composition differs across clusters

C0 and C2 have higher average Total\_Children values, while C1 and C3 have considerably fewer children on average.

### 6\. Store purchasing remains important

C1 and C3 have particularly high store-purchase activity, showing that the high-spending segments are not exclusively digital customers.

📣 14. Segment-Specific Marketing Strategy
==========================================

SegmentPrimary BehaviourRecommended Focus**C0 – Family-Oriented Lower-Spend**High browsing, low response and spendingImprove conversion through family offers and retargeting**C1 – High-Spending Multi-Channel**High income and spending across channelsLoyalty, repeat purchases and cross-selling**C2 – Digital-Engaged Lower-Spend**Highest website visits but lowest spendingDigital conversion, personalised offers and retargeting**C3 – High-Value Highly Responsive**High spending and strongest campaign responseRetention, premium service and high-value cross-selling

🔄 15. End-to-End Machine Learning Pipeline
===========================================

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML                    `CUSTOMER DATA                           │                           ▼                Data Exploration & Profiling                           │                           ▼                  Missing Value Treatment                    (Income → Median)                           │                           ▼                    Feature Engineering               ┌───────────┼────────────┐               ▼           ▼            ▼              Age      Customer Tenure  Total Spending                                        │                                    Total Children                           │                           ▼                    Feature Transformation               ┌───────────┴────────────┐               ▼                        ▼        Categorical Encoding       Feature Selection         One-Hot Encoding               │               ▼                    Outlier Treatment                      2240 → 2236                           │                           ▼                      Standard Scaling                           │                           ▼                           PCA                    18 Features → 3 PCs                           │               ┌───────────┴─────────────┐               ▼                         ▼         K-Means Clustering       Agglomerative Clustering            K = 4                  K = 4, Ward               │                         │               └───────────┬─────────────┘                           ▼                    Cluster Analysis                           │                           ▼                Customer Segmentation                    C0 | C1 | C2 | C3                           │                           ▼               Business Insights & Actions`

🧪 16. Model Summary
====================

ComponentImplementationLearning Type**Unsupervised Learning**Primary TaskCustomer SegmentationK-Means**4 clusters, random\_state=42**Agglomerative Clustering**4 clusters, Ward linkage**Cluster Selection**Elbow/Knee analysis**Silhouette AnalysisK = 2 to 10ScalingStandardScalerDimensionality ReductionPCAPCA Components Used for Clustering3Final Cluster CharacterisationAgglomerative labelsFinal Segments**C0, C1, C2, C3**

📊 17. Visual Analysis Performed
================================

The notebook includes visual analysis covering:

*   Pair plots for selected numerical variables
    
*   2D PCA visualisation
    
*   3D PCA visualisation
    
*   Correlation heatmap
    
*   Elbow/WCSS curve
    
*   Silhouette-score curve
    
*   Combined WCSS and silhouette plot
    
*   3D K-Means cluster visualisation
    
*   3D Agglomerative cluster visualisation
    
*   Cluster-size count plot
    
*   Income vs. Total Spending cluster scatter plot
    

These visualisations support both the technical clustering process and the interpretation of customer segments.

⚠️ 18. Interpretation Notes
===========================

The cluster labels **C0–C3 are algorithm-generated identifiers** and should not be interpreted as ordinal rankings.

The descriptive names used in this report — such as **Family-Oriented Lower-Spend Customers** and **High-Value Highly Responsive Customers** — are business interpretations derived from the cluster-level averages in the notebook.

The final cluster profiles are based on **Agglomerative Clustering labels**, while K-Means was also implemented using the same four-cluster structure.

The notebook establishes **K = 4 through the Elbow/Knee analysis**. Although silhouette scores were evaluated as an additional diagnostic, the highest silhouette value in the tested range was not at K=4. Therefore, the results should not claim that K=4 was selected because it maximised the silhouette score.

🚀 19. Business Impact
======================

The segmentation provides ShopSense with a data-driven framework for moving from generic customer communication toward more targeted strategies.

Instead of treating every customer identically, the business can differentiate its approach according to:

*   Customer spending
    
*   Income
    
*   Digital engagement
    
*   Store purchasing behaviour
    
*   Campaign responsiveness
    
*   Family composition
    
*   Customer tenure
    

This can support more relevant marketing campaigns, better allocation of promotional resources, stronger customer engagement, and improved retention of high-value customers.

✅ 20. Conclusion
================

The **ShopSense – E-Commerce Customer Segmentation Using Machine Learning** project demonstrates the application of unsupervised learning to identify meaningful customer groups from e-commerce data.

After data cleaning, feature engineering, outlier treatment, categorical encoding, standardisation, and PCA, both **K-Means** and **Agglomerative Hierarchical Clustering** were applied using four clusters. The Elbow/Knee analysis identified **K = 4** as the selected cluster count.

The final Agglomerative-clustering analysis identified four distinct behavioural profiles:

*   **C0 – Family-Oriented Lower-Spend Customers**
    
*   **C1 – High-Spending Multi-Channel Customers**
    
*   **C2 – Digital-Engaged Lower-Spend Customers**
    
*   **C3 – High-Value Highly Responsive Customers**
    

The analysis shows that customer value, purchasing channels, digital engagement, household composition, and campaign response differ substantially across these groups. These findings provide a practical basis for **segment-specific marketing, personalised engagement, customer retention, and more effective use of marketing resources**.

> **ShopSense — Turning Customer Data into Meaningful Customer Insights.**

🛠️ Technologies Used
---------------------

*   Python
    
*   Pandas
    
*   NumPy
    
*   Scikit-learn
    
*   Matplotlib
    
*   Seaborn
    
*   K-Means Clustering
    
*   Agglomerative Hierarchical Clustering
    
*   PCA
    
*   StandardScaler
    
*   Kneed / KneeLocator
    
*   Jupyter Notebook
    

📁 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ShopSense/  │  ├── data/  │   └── smartcart_customers.csv  │  ├── notebook/  │   └── ShopSense_Clustering.ipynb  │  ├── report/  │   └── ShopSense_Customer_Segmentation_Report.pdf  │  └── README.md   `

👨‍💻 Project Information
-------------------------

**Project Name:** ShopSense**Project Title:** E-Commerce Customer Segmentation Using Machine Learning**Learning Type:** Unsupervised Machine Learning**Algorithms:** K-Means & Agglomerative Hierarchical Clustering**Original Dataset:** 2,240 customers × 22 attributes**Post-Outlier Dataset:** 2,236 customers**Clustering Features:** 18 after encoding**PCA Representation:** 3 principal components**Selected Number of Clusters:** 4**Final Segments:** C0, C1, C2, C3

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML