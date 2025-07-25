from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f: # Load the scaler
    scaler = pickle.load(f)    

# Encode transaction types
type_encoding = {
    'PAYMENT': 1,
    'TRANSFER': 4,
    'CASH_OUT': 2,
    'DEBIT': 5,
    'CASH_IN': 3
}

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        tx_type = request.form['type'].strip().upper()

        encoded_type = type_encoding.get(tx_type, -1)
        if encoded_type == -1:
            return render_template('index.html', prediction_text="❌ Invalid transaction type.")

        # Final input to model
        features = [encoded_type, amount, oldbalanceOrg, newbalanceOrig]
        
        # Convert to a NumPy array and reshape for the scaler
        # The scaler expects a 2D array, even for a single sample
        input_data = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(input_data)

        prediction = model.predict(scaled_features)[0]
        result = "🚨 Fraudulent" if prediction == 1 else "✅ Legitimate"

        return render_template('index.html', prediction_text=f'Transaction is {result}.')

    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
