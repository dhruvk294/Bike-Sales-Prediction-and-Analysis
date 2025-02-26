import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Set Streamlit page title
st.title('Bike Sales Analysis and Prediction')

# Function to load dataset
@st.cache_data
def load_data():
    # Load the dataset and rename columns
    df = pd.read_csv('bike_sales_india.csv')
    df['Price Num'] = df['Price (INR)'].replace('[\₹,]', '', regex=True).astype(float)
    df['Age'] = 2025 - df['Year of Manufacture']
    df['Age_Price_Interaction'] = df['Age'] * df['Price Num']
    df['Mileage_Age_Interaction'] = df['Mileage (km/l)'] * df['Age']
    return df

# Check if dataset is already loaded in session state
if 'df' not in st.session_state:
    # Load the dataset once and store it in session state
    st.session_state['df'] = load_data()

# Retrieve the stored DataFrame
df = st.session_state['df']

# Display data preview
st.subheader('Data Preview')
st.write(df.head())

# Display basic information about the dataset
st.subheader('Dataset Info')
buffer = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes})
st.write(buffer)

# Identify categorical columns that need encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Perform One-Hot Encoding on all categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare the target and features
# Replaced 'Insurance Status' with 'Engine Capacity (cc)'
X = df_encoded.drop(columns=['Resale Price (INR)', 'Mileage (km/l)', 'Engine Capacity (cc)'])
y_price_cat = pd.qcut(df['Price Num'], q=3, labels=['Low', 'Mid', 'High'])

# Split data
X_train, X_test, y_train_price_cat, y_test_price_cat = train_test_split(X, y_price_cat, test_size=0.2, random_state=42)

# Model Training and Prediction
rf_price_cat = RandomForestClassifier(random_state=42)
rf_price_cat.fit(X_train, y_train_price_cat)
y_pred_price_cat_rf = rf_price_cat.predict(X_test)

gb_price_cat = GradientBoostingClassifier(random_state=42)
gb_price_cat.fit(X_train, y_train_price_cat)
y_pred_price_cat_gb = gb_price_cat.predict(X_test)

# Performance Metrics
price_cat_metrics = {
    'Model': ['Random Forest', 'Gradient Boosting'],
    'Accuracy': [accuracy_score(y_test_price_cat, y_pred_price_cat_rf), 
                 accuracy_score(y_test_price_cat, y_pred_price_cat_gb)],
    'Precision': [precision_score(y_test_price_cat, y_pred_price_cat_rf, average='weighted'), 
                  precision_score(y_test_price_cat, y_pred_price_cat_gb, average='weighted')],
    'Recall': [recall_score(y_test_price_cat, y_pred_price_cat_rf, average='weighted'), 
               recall_score(y_test_price_cat, y_pred_price_cat_gb, average='weighted')],
    'F1 Score': [f1_score(y_test_price_cat, y_pred_price_cat_rf, average='weighted'), 
                 f1_score(y_test_price_cat, y_pred_price_cat_gb, average='weighted')]
}

st.subheader('Price Category Classification Metrics')
st.write(pd.DataFrame(price_cat_metrics))

# Correlation Heatmap
st.subheader('Correlation Heatmap')
plt.figure(figsize=(15, 10))
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
st.pyplot(plt)

# Histogram for Engine Capacity
st.subheader('Distribution of Engine Capacity (cc)')
plt.figure(figsize=(10, 6))
sns.histplot(df['Engine Capacity (cc)'], bins=20, color='orange')
plt.title('Distribution of Engine Capacity (cc)')
plt.xlabel('Engine Capacity (cc)')
plt.ylabel('Frequency')
st.pyplot(plt)

# Confusion Matrix for Gradient Boosting
st.subheader('Confusion Matrix - Gradient Boosting Classifier')
cm = confusion_matrix(y_test_price_cat, y_pred_price_cat_gb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Mid', 'High'])
disp.plot(cmap='viridis')
st.pyplot(plt)
