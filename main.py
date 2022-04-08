from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())

    return render_template('index.html', locations = locations)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,sqft, bhk, bath]], columns=['location',  'total_sqft', 'bhk', 'bath'])
    prediction = pipe.predict(input)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug = True, port = 5001)