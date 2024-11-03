from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, template_folder='src')

# Load your model using joblib
saved_model = joblib.load('Diamond-Price-Prediction-Model-Project-Final.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve form data
        new_data = {
            'carat': [float(request.form['carat'])],
            'depth': [float(request.form['depth'])],
            'table': [float(request.form['table'])],
            'x': [float(request.form['x'])],
            'y': [float(request.form['y'])],
            'z': [float(request.form['z'])],
            'cut': [request.form['cut']],
            'color': [request.form['color']],
            'clarity': [request.form['clarity']]
        }

        # Convert the dictionary to a DataFrame
        new_data_df = pd.DataFrame(new_data)

        # Make predictions
        predictions = saved_model.predict(new_data_df)

        return render_template('form.html', prediction_text=f'Predicted value: ${predictions[0]:.2f}')

    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)
