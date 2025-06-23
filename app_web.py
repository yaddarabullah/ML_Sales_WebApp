# 3. 🌐 app_web.py – Web App dengan Flask
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('sales_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ads = float(request.form['ads'])
    products = float(request.form['products'])
    customers = float(request.form['customers'])
    
    prediction = model.predict(np.array([[ads, products, customers]]))
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f"Estimated Daily Sales: Rp {output:,.0f}")

if __name__ == '__main__':
    app.run(debug=True)
