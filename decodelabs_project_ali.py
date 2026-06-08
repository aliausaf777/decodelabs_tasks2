# ================================================================
#   DECODELABS DATA SCIENCE INTERNSHIP — TASK PROJECT
#   Tasks Completed : Task 1 | Task 2 | Task 3
#   Dataset         : Titanic (loaded via seaborn)
#   Author          : Ali
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# TASK 1: DATA COLLECTION & DATASET UNDERSTANDING
# ================================================================
# GOAL: Collect or load a dataset and understand its structure.

print("=" * 60)
print("  TASK 1: DATA COLLECTION & DATASET UNDERSTANDING")
print("=" * 60)

# ── Load dataset ────────────────────────────────────────────────
df_raw = sns.load_dataset('titanic')
print("\n📦 Dataset: Titanic (891 passengers, 15 features)")
print("   Source  : Seaborn built-in dataset (original Kaggle data)")

# ── Shape ───────────────────────────────────────────────────────
print(f"\n🔷 Shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ── Columns and data types ──────────────────────────────────────
print("\n🔷 Column Names & Data Types:")
print("-" * 40)
print(df_raw.dtypes.to_string())

# ── First look ──────────────────────────────────────────────────
print("\n🔷 First 5 Rows (raw data preview):")
print("-" * 40)
print(df_raw.head().to_string())

# ── What the data represents ────────────────────────────────────
print("""
🔷 Dataset Description:
   The Titanic dataset records information about 891 passengers
   aboard the RMS Titanic, which sank on April 15, 1912.
   Each row = one passenger.

   Columns:
   survived   → 0=No, 1=Yes  (target / outcome)
   pclass     → Ticket class (1=1st, 2=2nd, 3=3rd)
   sex        → Gender
   age        → Age in years
   sibsp      → # siblings / spouses aboard
   parch      → # parents / children aboard
   fare       → Ticket fare paid (£)
   embarked   → Port: S=Southampton, C=Cherbourg, Q=Queenstown
   class      → Text version of pclass (redundant)
   who        → man / woman / child (redundant)
   adult_male → Boolean (redundant)
   deck       → Cabin deck (mostly missing)
   embark_town→ Full city name (redundant)
   alive      → yes/no version of survived (redundant)
   alone      → Boolean — traveling alone
""")

# ── Missing values ──────────────────────────────────────────────
print("🔷 Missing Values:")
print("-" * 40)
missing = df_raw.isnull().sum()
pct     = (missing / len(df_raw) * 100).round(2)
mv_df   = pd.DataFrame({'Count': missing, 'Percentage (%)': pct})
print(mv_df[mv_df['Count'] > 0].to_string())


# ================================================================
# TASK 2: DATA CLEANING & PREPROCESSING
# ================================================================
# GOAL: Prepare the dataset for analysis by cleaning and organizing data.

print("\n\n" + "=" * 60)
print("  TASK 2: DATA CLEANING & PREPROCESSING")
print("=" * 60)

df = df_raw.copy()
print(f"\n🔷 Before Cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Step 1: Remove duplicates ───────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n✅ Step 1 | Duplicates removed   : {before - len(df)}")

# ── Step 2: Handle missing values ──────────────────────────────
# age → median (robust against skewed distribution)
median_age = df['age'].median()
df['age'].fillna(median_age, inplace=True)
print(f"✅ Step 2 | age  → filled with median ({median_age:.1f})")

# embarked → mode (most frequent port)
mode_emb = df['embarked'].mode()[0]
df['embarked'].fillna(mode_emb, inplace=True)
print(f"           embarked → filled with mode ('{mode_emb}')")

# deck → drop (77%+ missing, not recoverable)
df.drop(columns=['deck'], inplace=True)
print("           deck → dropped (>77% missing, unrecoverable)")

# ── Step 3: Drop redundant columns ─────────────────────────────
redundant = ['who', 'adult_male', 'embark_town', 'alive', 'alone', 'class']
df.drop(columns=redundant, inplace=True)
print(f"✅ Step 3 | Redundant columns dropped: {redundant}")

# ── Step 4: Encode categorical columns ─────────────────────────
df['sex']      = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print("✅ Step 4 | Encoding: sex (male=0, female=1) | embarked (S=0, C=1, Q=2)")

# ── Step 5: Fix data types ──────────────────────────────────────
df['pclass']   = df['pclass'].astype(int)
df['survived'] = df['survived'].astype(int)
df['sex']      = df['sex'].astype(int)
print("✅ Step 5 | Data types corrected (pclass, survived, sex → int)")

print(f"\n🔷 After Cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n🔷 Cleaned Data Preview:")
print(df.head().to_string())

remaining_nulls = df.isnull().sum().sum()
if remaining_nulls == 0:
    print("\n✅ No missing values remaining — dataset is clean!")
else:
    print(f"\n⚠️  Remaining nulls: {remaining_nulls}")


# ================================================================
# TASK 3: EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================
# GOAL: Analyze the dataset to discover patterns and trends.

print("\n\n" + "=" * 60)
print("  TASK 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# ── Basic statistics ────────────────────────────────────────────
print("\n🔷 Descriptive Statistics:")
print("-" * 40)
print(df.describe().round(2).to_string())

# ── Survival rate ───────────────────────────────────────────────
surv_rate = df['survived'].mean() * 100
print(f"\n🔷 Overall Survival Rate: {surv_rate:.1f}%")

# ── Survival by gender ──────────────────────────────────────────
print("\n🔷 Survival Rate by Gender:")
g = df_raw.groupby('sex')['survived'].mean() * 100
for idx, val in g.items():
    print(f"   {idx.capitalize():8s}: {val:.1f}%")

# ── Survival by class ───────────────────────────────────────────
print("\n🔷 Survival Rate by Passenger Class:")
c = df_raw.groupby('pclass')['survived'].mean() * 100
class_map = {1: '1st', 2: '2nd', 3: '3rd'}
for idx, val in c.items():
    print(f"   {class_map[idx]} Class: {val:.1f}%")

# ── Age stats ───────────────────────────────────────────────────
print(f"\n🔷 Age Statistics:")
print(f"   Min    : {df['age'].min():.1f}")
print(f"   Max    : {df['age'].max():.1f}")
print(f"   Mean   : {df['age'].mean():.1f}")
print(f"   Median : {df['age'].median():.1f}")
print(f"   Std Dev: {df['age'].std():.1f}")

# ── Outlier detection (IQR method on Fare) ──────────────────────
Q1, Q3 = df['fare'].quantile(0.25), df['fare'].quantile(0.75)
IQR    = Q3 - Q1
lower  = Q1 - 1.5 * IQR
upper  = Q3 + 1.5 * IQR
outliers = df[(df['fare'] < lower) | (df['fare'] > upper)]
print(f"\n🔷 Outlier Detection — Fare (IQR Method):")
print(f"   Q1={Q1:.2f} | Q3={Q3:.2f} | IQR={IQR:.2f}")
print(f"   Lower bound={lower:.2f} | Upper bound={upper:.2f}")
print(f"   Fare outliers: {len(outliers)} passengers ({len(outliers)/len(df)*100:.1f}%)")
print(f"   Highest fare paid: £{df['fare'].max():.2f}")

# ── Correlation ─────────────────────────────────────────────────
print("\n🔷 Correlation with Survival:")
corr = df.corr()['survived'].drop('survived').sort_values(ascending=False)
print(corr.round(3).to_string())

# ── Key findings ────────────────────────────────────────────────
print("""
🔷 KEY FINDINGS SUMMARY:
─────────────────────────────────────────────────────
1. Only ~38-41% of passengers survived the disaster.
2. Gender was the strongest factor: females survived at
   ~74% vs males at ~19%. The "women and children first"
   protocol was clearly followed.
3. Passenger class mattered significantly:
   1st class: 63% | 2nd class: 47% | 3rd class: 24%
   Wealthier passengers had better access to lifeboats.
4. Age: average passenger was ~30 years old.
   Children (age < 10) had relatively higher survival.
5. Fare: wide range (£0 to £512). 102 outliers detected.
   Higher fares correlated with better survival (class effect).
6. Southampton was the most common embarkation point (~72%).
─────────────────────────────────────────────────────
""")

# ================================================================
# VISUALIZATIONS — Saved as PNG
# ================================================================
print("📊 Generating visualizations...")

BG = '#f5f5e8'; GREEN = '#4a7c59'; LIGHT = '#a8c5a0'; ORANGE = '#e07b39'
fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.98, 'Titanic Dataset — EDA Report',
         ha='center', fontsize=22, fontweight='bold', color=GREEN)
fig.text(0.5, 0.965,
         'Tasks 1 + 2 + 3  |  Decodelabs Data Science Internship',
         ha='center', fontsize=11, color='#666')

positions = [
    [0.07, 0.73, 0.38, 0.21],
    [0.57, 0.73, 0.38, 0.21],
    [0.07, 0.47, 0.38, 0.21],
    [0.57, 0.47, 0.38, 0.21],
    [0.07, 0.21, 0.38, 0.21],
    [0.57, 0.21, 0.38, 0.21],
]
axes = [fig.add_axes(p) for p in positions]
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color('#ccc')

# 1. Survival count
sc = df['survived'].value_counts()
b1 = axes[0].bar(['Not Survived','Survived'],[sc[0],sc[1]],
                  color=[ORANGE,GREEN],width=0.5,edgecolor='white',linewidth=1.5)
axes[0].set_title('1. Survival Count', fontsize=13, fontweight='bold', color=GREEN, pad=8)
axes[0].set_ylabel('Passengers', color='#444', fontsize=10)
axes[0].set_ylim(0, 620)
for b in b1:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+5,
                 str(int(b.get_height())), ha='center', fontweight='bold', fontsize=11)

# 2. Survival by gender
gd = df_raw.groupby('sex')['survived'].mean()*100
b2 = axes[1].bar(gd.index, gd.values, color=[LIGHT,GREEN],
                  width=0.4, edgecolor='white', linewidth=1.5)
axes[1].set_title('2. Survival Rate by Gender', fontsize=13, fontweight='bold', color=GREEN, pad=8)
axes[1].set_ylabel('Survival Rate (%)', color='#444', fontsize=10)
axes[1].set_ylim(0, 100)
for b in b2:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                 f'{b.get_height():.1f}%', ha='center', fontweight='bold', fontsize=11)

# 3. Survival by class
cd = df_raw.groupby('pclass')['survived'].mean()*100
b3 = axes[2].bar(['1st','2nd','3rd'], cd.values,
                  color=[GREEN,LIGHT,ORANGE], width=0.5, edgecolor='white', linewidth=1.5)
axes[2].set_title('3. Survival Rate by Class', fontsize=13, fontweight='bold', color=GREEN, pad=8)
axes[2].set_ylabel('Survival Rate (%)', color='#444', fontsize=10)
axes[2].set_ylim(0, 90)
for b in b3:
    axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f'{b.get_height():.1f}%', ha='center', fontweight='bold', fontsize=11)

# 4. Age histogram
axes[3].hist(df_raw[df_raw['survived']==0]['age'].dropna(),
             bins=25, alpha=0.7, color=ORANGE, label='Not Survived', edgecolor='white')
axes[3].hist(df_raw[df_raw['survived']==1]['age'].dropna(),
             bins=25, alpha=0.7, color=GREEN, label='Survived', edgecolor='white')
axes[3].set_title('4. Age Distribution by Survival', fontsize=13, fontweight='bold', color=GREEN, pad=8)
axes[3].set_xlabel('Age', color='#444', fontsize=10)
axes[3].set_ylabel('Count', color='#444', fontsize=10)
axes[3].legend(fontsize=9)

# 5. Fare boxplot by class
bp = axes[4].boxplot(
    [df_raw[df_raw['pclass']==i]['fare'].dropna().values for i in [1,2,3]],
    labels=['1st','2nd','3rd'], patch_artist=True,
    medianprops=dict(color='white', linewidth=2))
for patch, col in zip(bp['boxes'], [GREEN,LIGHT,ORANGE]):
    patch.set_facecolor(col); patch.set_alpha(0.85)
axes[4].set_title('5. Fare Distribution by Class', fontsize=13, fontweight='bold', color=GREEN, pad=8)
axes[4].set_ylabel('Fare (£)', color='#444', fontsize=10)

# 6. Embarked pie
ep = df_raw['embarked'].value_counts()
port_lbl = {'S':'Southampton','C':'Cherbourg','Q':'Queenstown'}
axes[5].pie(ep.values, labels=[port_lbl.get(x,x) for x in ep.index],
            autopct='%1.1f%%', colors=[GREEN,LIGHT,ORANGE],
            startangle=140, textprops={'fontsize':9},
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[5].set_title('6. Embarkation Port Distribution', fontsize=13, fontweight='bold', color=GREEN, pad=8)

fig.text(0.5, 0.03,
         'Key: Survival 38–41% overall | Female 74% vs Male 19% | '
         '1st Class 63% vs 3rd 24% | Avg age ~30 | 102 fare outliers',
         ha='center', fontsize=9, color='#555', style='italic')

plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("✅ Visualizations saved as 'eda_visualizations.png'")

print("\n" + "=" * 60)
print("  ALL 3 TASKS COMPLETED SUCCESSFULLY ✅")
print("  Task 1: Data Collection & Dataset Understanding")
print("  Task 2: Data Cleaning & Preprocessing")
print("  Task 3: Exploratory Data Analysis (EDA)")
print("=" * 60)
