from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
car_df = pd.read_csv(r"D:\Machine Learning\Car_price_prediction\cleaned_car.csv")


@app.route('/')
def index():
    companies = sorted(car_df['company'].unique())
    companies.insert(0,"Select Company:")
    car_model = sorted(car_df['name'].unique())
    year = sorted(car_df['year'].unique())
    fuel_type = car_df['fuel_type'].unique()

    return render_template('index.html',companies=companies,car_models=car_model,years = year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_Model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)