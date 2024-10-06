from flask import Flask, render_template, request # type: ignore
import joblib # type: ignore
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        venue = int(request.form['venue'])
        bat_team = int(request.form['bat_team'])
        bowl_team = int(request.form['bowl_team'])
        batsman = int(request.form['batsman'])
        bowler = int(request.form['bowler'])        

        
        features = np.array([[venue, bat_team, bowl_team, batsman, bowler]])

        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Predicted Total Runs: {round(prediction[0], 2)}")

if __name__ == "__main__":
    app.run(debug=True)
