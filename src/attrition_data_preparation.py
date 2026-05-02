import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter


# ------------------------------
# 1. Load Dataset & Copy
# ------------------------------
df_original = pd.read_excel('D:/Users/phisi/OneDrive - Thammasat University/Coding/DX224/Project/Data/employee_attrition_dataset_10000.xlsx')
df = df_original.copy()

# ------------------------------
# 2. Display Settings
# ------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)

# ------------------------------
# 3. Initial Data Overview
# ------------------------------
print("📌 First 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nDescriptive Statistics (Numeric):")
print(df.describe())

print("\nDescriptive Statistics (Categorical):")
print(df.describe(include='object'))

print("\nUnique values of categorical columns:")
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

# ------------------------------
# 4. Visualize Distributions
# ------------------------------

# 4.1 Categorical Variable Distribution
for i in range(0, len(categorical_cols), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j in range(2):
        if i + j < len(categorical_cols):
            col = categorical_cols[i + j]
            sns.countplot(data=df, x=col, hue='Attrition', palette='pastel', ax=axes[j])
            axes[j].set_title(f'Distribution of {col}')
            axes[j].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# 4.2 Numerical Variable Distribution
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for i in range(0, len(numerical_cols), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j in range(2):
        if i + j < len(numerical_cols):
            col = numerical_cols[i + j]
            sns.histplot(data=df, x=col, hue='Attrition', kde=True, bins=30, palette='Set2', ax=axes[j])
            axes[j].set_title(f'Distribution of {col}')
            axes[j].set_xlabel(col)
    plt.tight_layout()
    plt.show()


# ------------------------------
# 1. Accuracy – ตรวจหาค่าที่ผิดปกติ (Outliers)
# ------------------------------
# Boxplot + Outlier Check with Label for Each Graph
# ------------------------------
print("[1] Accuracy: Outlier Detection")

# 1. เลือกเฉพาะตัวแปรเชิงตัวเลขที่ไม่ใช่ binary
non_binary_cols = [col for col in numerical_cols if df[col].nunique() > 2]

# 2. ฟังก์ชันตรวจ outliers ด้วย IQR
def has_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)].count() > 0

for i in range(0, len(non_binary_cols), 2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 แถว 2 คอลัมน์

    for j in range(2):
        if i + j < len(non_binary_cols):
            col = non_binary_cols[i + j]
            status = "Outliers detected" if has_outliers(df[col]) else "No outliers"

            sns.boxplot(data=df, x=col, color="lightblue", ax=axes[j])
            axes[j].set_title(f'Boxplot: {col}  •  {status}')

    plt.tight_layout()
    plt.show()



# ------------------------------
# 2. Completeness – ตรวจหาค่าที่หายไป (Missing Values)
# ------------------------------
print("\n [2] Completeness: Missing Values")

missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing (null) values found.")

# ค่าที่น่าสงสัย เช่น na, null, ?, -, n/a, none, missing
suspicious_values = ['na', 'n/a', 'null', 'none', '-', '', '?', 'missing']
for col in df.select_dtypes(include='object'):
    suspicious_count = df[col].astype(str).str.lower().isin(suspicious_values).sum()
    if suspicious_count > 0:
        print(f"Suspicious values in '{col}': {suspicious_count}")
    else:
        print(f"No suspicious strings in '{col}'")


# ------------------------------
# 3. Consistency – ตรวจสอบค่าที่ไม่ตรงกัน/สะกดผิด
# ------------------------------
print("\n[3] Consistency: Inconsistent Categorical Values")

for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    print(f"{col} - Unique Values: {list(unique_vals)}")


# ------------------------------
# 4. Timeliness – ตรวจสอบความเป็นปัจจุบัน (if available)
# ------------------------------
print("\n[4] Timeliness: Date Validation")

# ไม่มีข้อมูลวัน/เวลาใน dataset นี้ จึงไม่สามารถวัด Timeliness ได้
print("No date/time columns present → Timeliness cannot be evaluated.")


# ------------------------------
# 5. Uniqueness – ตรวจสอบค่า ID และข้อมูลซ้ำ
# ------------------------------
print("\n[5] Uniqueness: Duplicates and ID Checks")

# ตรวจแถวซ้ำ
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}" if duplicates > 0 else "No duplicate rows found.")

# ตรวจ Employee_ID ซ้ำ
if 'Employee_ID' in df.columns:
    unique_ids = df['Employee_ID'].nunique()
    total_ids = df.shape[0]
    if unique_ids < total_ids:
        print(f"Employee_ID not unique → {total_ids - unique_ids} duplicates found")
    else:
        print("Employee_ID is unique.")
else:
    print("Column 'Employee_ID' not found")

print("\n📌 [3.1] Feature Selection – Correlation with Attrition")

df_encoded = df.copy()

# Binary Encoding
if 'Attrition' in df_encoded.columns:
    df_encoded['Attrition'] = df_encoded['Attrition'].map({'Yes': 1, 'No': 0})
if 'Overtime' in df_encoded.columns:
    df_encoded['Overtime'] = df_encoded['Overtime'].map({'Yes': 1, 'No': 0})
if 'Gender' in df_encoded.columns:
    df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})

# One-hot Encoding
df_encoded = pd.get_dummies(df_encoded, drop_first=True, dtype=int)
print(df_encoded.head())

# (1) Correlation: เลือก Top 15 ฟีเจอร์ที่สัมพันธ์กับ Attrition
cor_matrix = df_encoded.drop(columns=['Employee_ID']).corr()
corr_top_features = cor_matrix['Attrition'].abs().sort_values(ascending=False).drop('Attrition').head(15)
corr_feature_set = set(corr_top_features.index)

# (2) Random Forest Feature Importance
X_all = df_encoded.drop(columns=['Employee_ID', 'Attrition'])
y_all = df_encoded['Attrition']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_all, y_all)

importance_df = pd.DataFrame({
    'Feature': X_all.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

rf_top_features = importance_df.head(15)['Feature']
rf_feature_set = set(rf_top_features)

# (3) ฟีเจอร์ที่อยู่ในทั้ง 2 กลุ่ม
selected_features = list(corr_feature_set | rf_feature_set)
print("Features selected from both Correlation & Random Forest:\n", selected_features)

# (4) เตรียมชุดข้อมูลสุดท้าย
df_selected_combined = df_encoded[selected_features + ['Attrition']]
X = df_selected_combined.drop('Attrition', axis=1)
y = df_selected_combined['Attrition']

# ดึง top 15 จากทั้งสองวิธี
top_corr = cor_matrix['Attrition'].abs().sort_values(ascending=False).drop('Attrition').head(15)
top_corr_df = top_corr.reset_index().rename(columns={'index': 'Feature', 'Attrition': 'Score'})
top_corr_df['Method'] = 'Correlation'

top_rf = importance_df.head(15)
top_rf_df = top_rf.rename(columns={'Importance': 'Score'})
top_rf_df['Method'] = 'Random Forest'

# รวมข้อมูล
compare_df = pd.concat([top_corr_df, top_rf_df])

# วาดกราฟ
plt.figure(figsize=(12, 6))
sns.barplot(data=compare_df, x='Score', y='Feature', hue='Method', dodge=True, palette=['#5DADE2', '#F5B041'])
plt.title('Comparison of Top 15 Features: Correlation vs Random Forest')
plt.xlabel('Score (Correlation / Importance)')
plt.ylabel('Feature')
plt.legend(title='Selection Method')
plt.tight_layout()
plt.show()

smote = SMOTE(random_state=42)
X_selected_resampled, y_selected_resampled = smote.fit_resample(X, y)

# ก่อนทำ SMOTE
original_counts = Counter(y)

# หลังทำ SMOTE
resampled_counts = Counter(y_selected_resampled)

# สร้าง DataFrame เพื่อเปรียบเทียบ
df_compare = pd.DataFrame({
    'Original': pd.Series(original_counts),
    'Resampled': pd.Series(resampled_counts)
})
print(df_compare)

# วาดกราฟเปรียบเทียบ
df_compare.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
plt.title('Class Distribution Before and After SMOTE')
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_selected_resampled_scaled = scaler.fit_transform(X_selected_resampled)

# เปรียบเทียบข้อมูลก่อนและหลังการ Scaling
df_comparison = pd.DataFrame({
    'Original (Before Scaling)': X_selected_resampled.iloc[:, :5].mean(),  # ใช้แค่ 5 ฟีเจอร์แรกในตัวอย่าง
    'Scaled (After Scaling)': X_selected_resampled_scaled[:, :5].mean()  # ใช้แค่ 5 ฟีเจอร์แรกหลัง Scaling
})

# กราฟแสดงการเปรียบเทียบค่าเฉลี่ยก่อนและหลัง Scaling
df_comparison.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Comparison of Feature Means Before and After Scaling')
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# เปรียบเทียบ Standard Deviation
df_comparison_std = pd.DataFrame({
    'Original (Before Scaling)': X_selected_resampled.iloc[:, :5].std(),
    'Scaled (After Scaling)': X_selected_resampled_scaled[:, :5].std()
})

# กราฟแสดงการเปรียบเทียบ Standard Deviation
df_comparison_std.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Comparison of Feature Standard Deviations Before and After Scaling')
plt.xlabel('Feature')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# สมมติว่า X_selected_resampled คือ DataFrame
X_before = pd.DataFrame(X_selected_resampled, columns=X.columns)
scaler = StandardScaler()
X_after = pd.DataFrame(scaler.fit_transform(X_selected_resampled), columns=X.columns)
print("📌 ก่อน Scaling:\n", X_before.describe().T[['mean', 'std']])
print("\n📌 หลัง Scaling:\n", X_after.describe().T[['mean', 'std']])

# สร้างสำเนาชุดข้อมูล
df_binned = X_selected_resampled.copy()

# Age
df_binned['Age_Bin'] = pd.cut(X_selected_resampled['Age'],
    bins=[17, 25, 35, 45, 60],
    labels=['Age=18-25', 'Age=26-35', 'Age=36-45', 'Age=46-60'])

# Years Since Last Promotion
df_binned['Years_Since_Last_Promotion_Bin'] = pd.cut(X_selected_resampled['Years_Since_Last_Promotion'],
    bins=[-1, 0, 2, 5, 10, 40],
    labels=['Promo=0', 'Promo=1-2', 'Promo=3-5', 'Promo=6-10', 'Promo=10+'])

# Monthly Income
df_binned['Monthly_Income_Bin'] = pd.cut(X_selected_resampled['Monthly_Income'],
    bins=[0, 30000, 60000, 100000, float('inf')],
    labels=['Income=<=30k', 'Income=30k-60k', 'Income=60k-100k', 'Income=100k+'])

# Training Hours Last Year
df_binned['Training_Hours_Last_Year_Bin'] = pd.cut(X_selected_resampled['Training_Hours_Last_Year'],
    bins=[-1, 10, 30, 60, float('inf')],
    labels=['Train=0-10', 'Train=11-30', 'Train=31-60', 'Train=60+'])

# Distance From Home
df_binned['Distance_From_Home_Bin'] = pd.cut(X_selected_resampled['Distance_From_Home'],
    bins=[-1, 5, 10, 20, float('inf')],
    labels=['Dist=0-5km', 'Dist=6-10km', 'Dist=11-20km', 'Dist=20km+'])

# Years at Company
df_binned['Years_at_Company_Bin'] = pd.cut(X_selected_resampled['Years_at_Company'],
    bins=[-1, 1, 5, 10, 20, float('inf')],
    labels=['YearsAtComp=0-1', 'YearsAtComp=2-5', 'YearsAtComp=6-10', 'YearsAtComp=11-20', 'YearsAtComp=20+'])

# Years in Current Role
df_binned['Years_in_Current_Role_Bin'] = pd.cut(X_selected_resampled['Years_in_Current_Role'],
    bins=[-1, 1, 3, 6, 10, float('inf')],
    labels=['CurrRole=0-1', 'CurrRole=2-3', 'CurrRole=4-6', 'CurrRole=7-10', 'CurrRole=10+'])

# Project Count
df_binned['Project_Count_Bin'] = pd.cut(X_selected_resampled['Project_Count'],
    bins=[-1, 2, 4, 6, float('inf')],
    labels=['Projects=<=2', 'Projects=3-4', 'Projects=5-6', 'Projects=7+'])

# Absenteeism
df_binned['Absenteeism_Bin'] = pd.cut(X_selected_resampled['Absenteeism'],
    bins=[-1, 2, 5, 10, float('inf')],
    labels=['Abs=0-2', 'Abs=3-5', 'Abs=6-10', 'Abs=11+'])

# Average Hours Worked
df_binned['Average_Hours_Worked_Per_Week_Bin'] = pd.cut(X_selected_resampled['Average_Hours_Worked_Per_Week'],
    bins=[0, 30, 45, 60, float('inf')],
    labels=['Hrs=<=30', 'Hrs=31-45', 'Hrs=46-60', 'Hrs=60+'])

# Hourly Rate
df_binned['Hourly_Rate_Bin'] = pd.cut(X_selected_resampled['Hourly_Rate'],
    bins=[0, 40, 60, 80, 100],
    labels=['Rate=<=40', 'Rate=41-60', 'Rate=61-80', 'Rate=81-100'])

# Work Life Balance
df_binned['Work_Life_Balance_Bin'] = pd.cut(X_selected_resampled['Work_Life_Balance'],
    bins=[0, 1, 2, 3, 4],
    labels=['WLB=Poor', 'WLB=Fair', 'WLB=Good', 'WLB=Excellent'])

# Relationship with Manager
df_binned['Relationship_with_Manager_Bin'] = pd.cut(X_selected_resampled['Relationship_with_Manager'],
    bins=[0, 2, 3, 4, 5],
    labels=['MgrRel=Poor', 'MgrRel=Average', 'MgrRel=Good', 'MgrRel=Excellent'])

# Job Involvement
df_binned['Job_Involvement_Bin'] = pd.cut(X_selected_resampled['Job_Involvement'],
    bins=[0, 2, 3, 4, 5],
    labels=['Inv=Low', 'Inv=Medium', 'Inv=High', 'Inv=Very High'])

# Job Satisfaction
df_binned['Job_Satisfaction_Bin'] = pd.cut(X_selected_resampled['Job_Satisfaction'],
    bins=[0, 1, 2, 3, 4, 5],
    labels=['JSat=Very Low', 'JSat=Low', 'JSat=Moderate', 'JSat=High', 'JSat=Very High'])

# Performance Rating
df_binned['Performance_Rating_Bin'] = pd.cut(X_selected_resampled['Performance_Rating'],
    bins=[0, 2, 3, 4, 5],
    labels=['Perf=Below Avg', 'Perf=Average', 'Perf=Good', 'Perf=Excellent'])

# Work Environment Satisfaction
df_binned['Work_Environment_Satisfaction_Bin'] = pd.cut(X_selected_resampled['Work_Environment_Satisfaction'],
    bins=[0, 1, 2, 3, 4],
    labels=['EnvSat=Low', 'EnvSat=Moderate', 'EnvSat=High', 'EnvSat=Very High'])

# Job Level
df_binned['Job_Level_Bin'] = pd.cut(X_selected_resampled['Job_Level'],
    bins=[0, 1, 2, 3, 4, 5],
    labels=['JobLevel=1', 'JobLevel=2', 'JobLevel=3', 'JobLevel=4', 'JobLevel=5'])

# Number of Companies Worked
df_binned['Number_of_Companies_Worked_Bin'] = pd.cut(X_selected_resampled['Number_of_Companies_Worked'],
    bins=[-1, 0, 2, 5, float('inf')],
    labels=['Companies=0', 'Companies=1-2', 'Companies=3-5', 'Companies=6+'])

# Gender (0 or 1) → ทำให้เป็นหมวดหมู่
df_binned['Gender_Bin'] = X_selected_resampled['Gender'].map({0: 'Female', 1: 'Male'})

# Job_Role_Manager → ใช้ label ตรง ๆ
df_binned['Job_Role_Manager_Bin'] = X_selected_resampled['Job_Role_Manager'].map({0: 'Not Manager', 1: 'Manager'})
df_binned['Job_Role_Assistant_Bin'] = X_selected_resampled['Job_Role_Assistant'].map({0: 'Not Assistant', 1: 'Assistant'})

# Department_HR, Department_Marketing
df_binned['Department_HR_Bin'] = X_selected_resampled['Department_HR'].map({0: 'Not HR', 1: 'HR'})
df_binned['Department_Marketing_Bin'] = X_selected_resampled['Department_Marketing'].map({0: 'Not Marketing', 1: 'Marketing'})

# Marital_Status_Married
df_binned['Marital_Status_Married_Bin'] = X_selected_resampled['Marital_Status_Married'].map({0: 'Not Married', 1: 'Married'})

df_binned['Attrition'] = y_selected_resampled.values
df_binned['Attrition'] = df_binned['Attrition'].map({1: 'Attrition_Yes', 0: 'Attrition_No'})




bin_cols = [col for col in df_binned.columns if col.endswith('_Bin')]
print(df_binned[bin_cols].head())


# ลดจำนวนแถวเหลือ 10,000 หรือ 5,000 ตัวอย่าง
df_sampled = df_binned.sample(n=10000, random_state=42)
print("✅ Sampling done!", df_sampled.shape)

transactions = df_sampled[bin_cols + ['Attrition']].astype(str).values.tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)

print("Shape of df_te:", df_te.shape)
print("Total unique items:", len(df_te.columns))
print(df_te.dtypes.value_counts())  # ควรมีแค่ bool

# รัน apriori
frequent_items = apriori(df_te, min_support=0.2, use_colnames=True)
# สร้าง rule ทั้งหมด
rules = association_rules(frequent_items, metric='lift', min_threshold=1.0)

# ตรวจสอบกฎเกี่ยวกับ Attrition
rules_target = rules[
    rules['consequents'].astype(str).str.contains('Attrition_Yes')
].sort_values(by='lift', ascending=False)


# จัดกลุ่มกฎตาม antecedents (แปลงเป็น set เพื่อดูความเป็น subset)
grouped_rules = []
used_sets = []

for _, row in rules_target.iterrows():
    antecedent_set = set(row['antecedents'])
    is_new = True
    for s in used_sets:
        if antecedent_set <= s or s <= antecedent_set:
            is_new = False
            break
    if is_new:
        grouped_rules.append(row)
        used_sets.append(antecedent_set)

# สร้าง DataFrame ใหม่เฉพาะกฎที่ไม่ซ้ำซ้อน
rules_dedup = pd.DataFrame(grouped_rules)

# คัดเฉพาะกฎที่ confidence >= 0.70 และ lift >= 1.4
highlighted_rules = rules_dedup[(rules_dedup['confidence'] >= 0.73) & (rules_dedup['lift'] >= 1.5)]

# จัดเรียงลำดับจาก confidence มากไปน้อย
highlighted_rules = highlighted_rules.sort_values(by='confidence', ascending=False).reset_index(drop=True)

# แสดงผล
print(highlighted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""
# แปลง X_selected_resampled_scaled เป็น DataFrame ก่อน export
scaled_df = pd.DataFrame(X_selected_resampled_scaled, columns=X.columns)
scaled_df['Attrition'] = y_selected_resampled.values

 scaled_df.to_csv("D:/Users/phisi/OneDrive - Thammasat University/Coding/DX224/Project/Data/scaled_resampled.csv", index=False)


binned_columns = [col for col in df_binned.columns if col.endswith("_Bin")] + ['Attrition']
df_binned_only = df_binned[binned_columns]
df_binned_only.insert(0, 'ID', range(1, 1 + len(df_binned_only)))

# Export ไป RapidMiner
df_binned_only.to_csv("df_binned_only.csv", index=False)

"""