#Utility code for data manipulation
import numpy as np
import pandas as pd

test_response = {
            'lead_time':123,
            'adr':245.50,
            'total_of_special_requests':2,
            'previous_cancellations':4,
            'deposit_type':'Non Refund',
            'country':'USA',
            'arrival_date_month':'April',
            'arrival_date_day_of_month':'21'
}

test_template = pd.read_csv('updated_modeling_dataset_final.csv')
test_template = test_template[:1]
test_template = test_template.drop(['is_canceled','Unnamed: 0'],axis=1)

class Utils:
    def __init__(self,template,response):
        self.template = template
        self.response = response


    def convert_to_pred(self):
        response_df = pd.DataFrame.from_dict(self.response,orient='index')
        response_df = response_df.T
        num_cols = ['lead_time','adr','total_of_special_requests','previous_cancellations']
        for i in num_cols:
            response_df[i]=response_df[i].astype('float')
        response_df = pd.get_dummies(response_df)
        result = self.template.append(response_df, ignore_index=True,sort=False)
        result.fillna(0.0,inplace=True)
        return result.tail(1)

Result = Utils(template=test_template,response=test_response)

final_format = Result.convert_to_pred()
