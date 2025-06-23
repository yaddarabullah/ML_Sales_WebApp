# 2. 🧠 train_model_app_web.py – Train Model ML
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Contoh data penjualan
data = {
    'Ads': [1, 2, 3, 4, 5, 6, 7],
    'Products_Displayed': [10, 20, 30, 40, 50, 60, 70],
    'Daily_Customers': [100, 150, 200, 250, 300, 350, 400],
    'Sales': [20000, 30000, 45000, 60000, 70000, 85000, 95000]  # in Rupiah
}

df = pd.DataFrame(data)

# X = fitur, y = target
X = df[['Ads', 'Products_Displayed', 'Daily_Customers']]
y = df['Sales']

# Model regresi linear
model = LinearRegression()
model.fit(X, y)

# Simpan model ke file .pkl
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)
