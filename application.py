from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app = application

### import randomforestclassifier and the scandardscaler pickle file
scalar = pickle.load(open("models/scaled.pkl",'rb'))
model = pickle.load(open("models/randonforest.pkl",'rb'))



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method =='POST':
        age=int(request.form.get("age"))
        job = int(request.form.get("job"))
        marital = int(request.form.get("marital"))
        education = int(request.form.get("education"))
        default = int(request.form.get("default"))
        balance = int(request.form.get("balance"))
        housing = int(request.form.get("housing"))
        loan = int(request.form.get("loan"))
        contact = int(request.form.get("contact"))
        day = int(request.form.get("day"))
        month = int(request.form.get("month"))
        duration = int(request.form.get("duration"))
        campaign = int(request.form.get("campaign"))
        pdays = int(request.form.get("pdays"))
        previous = int(request.form.get("previous"))
        poutcome = int(request.form.get("poutcome"))

        new_scaler=scaler.transform([age, job, marital, education, default, balance, housing,loan, contact, day, month, duration, campaign, pdays,previous, poutcome])
        result = model.predict(new_scaler)

        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")



