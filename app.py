from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import lightgbm as lgb

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
rf_model = joblib.load(os.path.join(MODELS_FOLDER, 'rf_model.pkl'))
gb_model = joblib.load(os.path.join(MODELS_FOLDER, 'gb_model.pkl'))
logreg_model = joblib.load(os.path.join(MODELS_FOLDER, 'logreg_model.pkl'))
tree_model = joblib.load(os.path.join(MODELS_FOLDER, 'tree_model.pkl'))
xgb_model = joblib.load(os.path.join(MODELS_FOLDER, 'xgb_model.pkl'))
lgb_model = lgb.Booster(model_file=os.path.join(MODELS_FOLDER, 'lgb_model.txt'))

# Define the columns to use globally
columns_to_use = ['SEX', 'INTUBED', 'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA',
                  'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
                  'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'AGE_INTUBED', 'PNEUMONIA_DIABETES']

# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

# Extract significant factors function with prioritization
def extract_significant_factors(model, X_scaled, input_row):
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif isinstance(model, lgb.Booster):
        feature_importances = model.feature_importance()
    else:
        feature_importances = np.abs(model.coef_[0])  # For models like Logistic Regression

    # Create a DataFrame for feature importances
    features_df = pd.DataFrame({
        'Feature': columns_to_use,
        'Importance': feature_importances
    })

    # Sort by importance, get top factors
    sorted_factors = features_df.sort_values(by='Importance', ascending=False)

    # Build a list of the significant factors for this specific patient
    significant_factors = []
    
    # Conditions to prioritize over 'SEX'
    prioritized_factors = ['AGE', 'DIABETES', 'PNEUMONIA', 'HIPERTENSION', 'COPD', 'ASTHMA', 'INMSUPR', 'OBESITY']

    # First prioritize other factors over 'SEX'
    for idx, row in sorted_factors.iterrows():
        feature = row['Feature']
        
        # Include prioritized conditions only if the patient has them or age is significant
        if feature in prioritized_factors and input_row[feature] == 1:
            significant_factors.append(feature)
        
        # Stop if we have found 3 significant factors
        if len(significant_factors) >= 3:
            break
    
    # Then, add 'SEX' if it's a significant factor and we have fewer than 3 factors
    if 'SEX' in sorted_factors['Feature'].values and input_row['SEX'] == 1 and len(significant_factors) < 3:
        significant_factors.append('SEX')

    # If no significant factors are found, return "No Significant Factors"
    if not significant_factors:
        significant_factors.append('No Significant Factors')

    # Return the relevant features for this patient
    return ', '.join(significant_factors)

# Preprocessing function for both manual and CSV
def preprocess_data(df):
    df = df.head(2000)  # Limit to first 2000 rows
    df['ALIVE_STATUS'] = np.where(df['DATE_DIED'] == '9999-99-99', 1, 0)
    df['PREGNANT'] = np.where(df['SEX'] == 2, 0, df['PREGNANT'])

    # Impute missing values
    imputer_pregnant = SimpleImputer(missing_values=98, strategy='most_frequent')
    df['PREGNANT'] = imputer_pregnant.fit_transform(df[['PREGNANT']])
    
    df['INTUBED'] = np.where(df['PATIENT_TYPE'] == 1, 0, df['INTUBED'])
    imputer_intubed = SimpleImputer(missing_values=99, strategy='most_frequent')
    df.loc[df['PATIENT_TYPE'] == 2, 'INTUBED'] = imputer_intubed.fit_transform(df[df['PATIENT_TYPE'] == 2][['INTUBED']])
    
    df['ICU'] = np.where(df['PATIENT_TYPE'] == 1, 0, df['ICU'])
    imputer_icu = SimpleImputer(missing_values=99, strategy='most_frequent')
    df.loc[df['PATIENT_TYPE'] == 2, 'ICU'] = imputer_icu.fit_transform(df[df['PATIENT_TYPE'] == 2][['ICU']])

    columns_with_placeholders = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 
                                 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    imputer_98 = SimpleImputer(missing_values=98, strategy='most_frequent')
    imputer_99 = SimpleImputer(missing_values=99, strategy='most_frequent')
    for col in columns_with_placeholders:
        df[col] = imputer_98.fit_transform(df[[col]])
    df['PNEUMONIA'] = imputer_99.fit_transform(df[['PNEUMONIA']])

    df['AGE_INTUBED'] = df['AGE'] * df['INTUBED']
    df['PNEUMONIA_DIABETES'] = df['PNEUMONIA'] * df['DIABETES']

    imputer = SimpleImputer(strategy='most_frequent')
    X = df[columns_to_use]
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, df

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# File upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            algorithm = request.form['algorithm']
            return redirect(url_for('process_csv', filename=filename, algorithm=algorithm))
    return render_template('upload.html')

# Process CSV and make predictions
@app.route('/process_csv/<filename>/<algorithm>')
def process_csv(filename, algorithm):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    X_processed, df_original = preprocess_data(df)

    if algorithm == 'Random Forest':
        model = rf_model
    elif algorithm == 'Gradient Boosting':
        model = gb_model
    elif algorithm == 'LightGBM':
        model = lgb_model
    elif algorithm == 'Decision Tree':
        model = tree_model
    elif algorithm == 'XGBoost':
        model = xgb_model
    else:
        model = logreg_model

    if algorithm == 'LightGBM':
        probabilities = model.predict(X_processed) * 100  # LightGBM uses predict()
    else:
        probabilities = model.predict_proba(X_processed)[:, 1] * 100

    # Ensure deceased patients have 100% risk
    df_original['Risk Percentage'] = np.where(df_original['ALIVE_STATUS'] == 0, 100, probabilities.round(2))

    df_original['Risk'] = np.where(df_original['ALIVE_STATUS'] == 0, 'High Risk', 
                                   np.where(probabilities > 60, 'High Risk', 
                                            np.where(probabilities > 40, 'Mid Risk', 'Low Risk')))
    
    df_original['Needs ICU'] = np.where(df_original['ALIVE_STATUS'] == 0, 
                                        np.where(df_original['ICU'] == 1, 'Admitted to ICU', 'Not Admitted to ICU'), 
                                        np.where(probabilities > 70, 'Yes', 'No'))

    # Identify significant factors
    significant_factors_list = []
    for i in range(X_processed.shape[0]):
        significant_factors = extract_significant_factors(model, pd.DataFrame(X_processed, columns=columns_to_use), df_original.iloc[i])
        significant_factors_list.append(significant_factors)

    df_original['Significant Factors'] = significant_factors_list

    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
    df_original.to_csv(result_filepath, index=False)

    # Pass enumerate to the template
    return render_template('result.html', results=df_original.head(2000).to_dict(orient='records'), enumerate=enumerate)

# Serve the result CSV file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Manual Entry route
@app.route('/manual', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data]).apply(pd.to_numeric, errors='coerce')

        input_df['AGE_INTUBED'] = input_df['AGE'] * input_df['INTUBED']
        input_df['PNEUMONIA_DIABETES'] = input_df['PNEUMONIA'] * input_df['DIABETES']

        imputer = SimpleImputer(strategy='most_frequent')
        X_manual = imputer.fit_transform(input_df[columns_to_use])
        scaler = StandardScaler()
        X_manual_scaled = scaler.fit_transform(X_manual)

        algorithm = request.form['algorithm']
        if algorithm == 'Random Forest':
            model = rf_model
        elif algorithm == 'Gradient Boosting':
            model = gb_model
        elif algorithm == 'LightGBM':
            model = lgb_model
        elif algorithm == 'Decision Tree':
            model = tree_model
        elif algorithm == 'XGBoost':
            model = xgb_model
        else:
            model = logreg_model

        if algorithm == 'LightGBM':
            probabilities = model.predict(X_manual_scaled) * 100  # LightGBM uses predict()
        else:
            probabilities = model.predict_proba(X_manual_scaled)[:, 1] * 100

        risk_percentage = probabilities[0].round(2)
        risk = 'High Risk' if risk_percentage > 70 else 'Mid Risk' if risk_percentage > 40 else 'Low Risk'
        icu_required = 'Yes' if risk == 'High Risk' else 'No'

        # Extract significant factors for manual entry
        significant_factors = extract_significant_factors(model, pd.DataFrame(X_manual_scaled, columns=columns_to_use), input_df.iloc[0])

        return render_template('manual_result.html', form_data=form_data, risk=risk, risk_percentage=risk_percentage, icu_required=icu_required, significant_factors=significant_factors)

    return render_template('manual.html')

if __name__ == '__main__':
    app.run(debug=True)
