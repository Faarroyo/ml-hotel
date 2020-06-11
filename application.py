#one page flask app
from flask import Flask,url_for,render_template,send_file,request,jsonify
from flask_bootstrap import Bootstrap
from utils import Utils
import pandas as pd
import numpy as np
import pickle
import joblib
app = Flask(__name__)
Bootstrap(app)

#FINALIZED MODEL IMPORTING
loaded_model = joblib.load('finalized_model_RF.pkl')
template = pd.read_csv('updated_modeling_dataset_final.csv')
template = template[:1]
template = template.drop(['is_canceled','Unnamed: 0'],axis=1)

#Routes
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/feat_select')
def feat_select():
    return render_template('feat_select.html')

@app.route('/getdata',methods=['GET'])
def download_data():
    return send_file('hotel_bookings.csv',
                     mimetype='text/csv',
                     attachment_filename='hotel_bookings.csv',
                     as_attachment=True)

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        keys = ['lead_time',
                    'adr',
                    'total_of_special_requests',
                    'previous_cancellations',
                    'deposit_type',
                    'country',
                    'arrival_date_month',
                    'arrival_date_day_of_month']

        values = [x for x in request.form.values()]
        response = dict(zip(keys,values))
        formatting = Utils(template=template,response=response)
        final_format = formatting.convert_to_pred()
        prediction = loaded_model.predict(final_format)
        if prediction == 1:
            output="Customer Will Cancel"
        else:
            output="Customer Will Not Cancel"
    except:
        output="Data Entry Error"

    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
