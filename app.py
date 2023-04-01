from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('Iris.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    pred=model.predict([[data1,data2,data3,data4]])[0]
    return render_template('after.html',data=pred)

@app.route('/tq')
def tq():
    return render_template('tq.html')
if __name__=='__main__':
    app.run(debug=True)

