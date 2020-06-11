from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import Utils
import joblib
import pandas as pd
import numpy as np

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
#Loading our final dataset
df = pd.read_csv('updated_modeling_dataset_final.csv')

#Split data for training
X = df.drop(['is_canceled','Unnamed: 0'],axis=1)
y = df['is_canceled']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y)

#Fit Model
model = RandomForestClassifier(max_depth=50,max_features=0.9,min_samples_split=5,class_weight={0:2,1:4})

model.fit(X_train,y_train)


test = Utils(template = X_train[:1],response=test_response)

test_results = test.convert_to_pred()

prediction = model.predict(test_results)

if prediction == 1:
    print(prediction)
elif prediction == 0:
    print(prediction)
else:
    print("Could not compute")
#Export Model
joblib.dump(model,filename='finalized_model_RF.pkl',compress=5)
