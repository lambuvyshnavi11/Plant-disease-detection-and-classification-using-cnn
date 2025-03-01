# Importing essential libraries and modules
import os
from flask import Flask, render_template, request, Markup, redirect
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model

import CNN
import torchvision.transforms.functional as TF
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import torch
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from sklearn.utils import resample
# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve

# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline

# Tuning
from sklearn.model_selection import GridSearchCV

# Feature Extraction
from sklearn.feature_selection import RFE

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder



# Ensembles
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model



disease_info = pd.read_csv('Data/disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('Data/supplement_info.csv',encoding='cp1252')

disease_model = CNN.CNN(39)
disease_model.load_state_dict(torch.load("models/plant_disease_model_1_latest.pt"))
disease_model.eval()





# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = disease_model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Home Page'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            # Loading crop recommendation using Random Forest model

            warnings.filterwarnings('ignore')

            sns.set_style("whitegrid", {'axes.grid': False})
            pio.templates.default = "plotly_white"

            ################################################################################
            #                                                                              #
            #                            Analyze Data                                      #
            #                                                                              #
            ################################################################################
            def explore_data(df):
                print("Number of Instances and Attributes:", df.shape)
                print('\n')
                print('Dataset columns:', df.columns)
                print('\n')
                print('Data types of each columns: ', df.info())

            ################################################################################
            #                                                                              #
            #                      Checking for Duplicates                                 #
            #                                                                              #
            ################################################################################
            def checking_removing_duplicates(df):
                count_dups = df.duplicated().sum()
                print("Number of Duplicates: ", count_dups)
                if count_dups >= 1:
                    df.drop_duplicates(inplace=True)
                    print('Duplicate values removed!')
                else:
                    print('No Duplicate values')

            ################################################################################
            #                                                                              #
            #                Split Data to Training and Validation set                     #
            #                                                                              #
            ################################################################################
            def read_in_and_split_data(data, target):
                X = data.drop(target, axis=1)
                y = data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                return X_train, X_test, y_train, y_test

            ################################################################################
            #                                                                              #
            #                        Spot-Check Algorithms                                 #
            #                                                                              #
            ################################################################################

            def ensemblemodels():
                ensembles = []

                ensembles.append(('RF', RandomForestClassifier()))

                return ensembles

            ################################################################################
            #                                                                              #
            #                 Spot-Check Normalized Models                                 #
            #                                                                              #
            ################################################################################
            def NormalizedModel(nameOfScaler):
                if nameOfScaler == 'standard':
                    scaler = StandardScaler()
                elif nameOfScaler == 'minmax':
                    scaler = MinMaxScaler()
                elif nameOfScaler == 'normalizer':
                    scaler = Normalizer()
                elif nameOfScaler == 'binarizer':
                    scaler = Binarizer()

                pipelines = []

                pipelines.append(
                    (nameOfScaler + 'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier())])))

                return pipelines

            ################################################################################
            #                                                                              #
            #                           Train Model                                        #
            #                                                                              #
            ################################################################################
            def fit_model(X_train, y_train, models):
                # Test options and evaluation metric
                num_folds = 10
                scoring = 'accuracy'

                results = []
                names = []
                for name, model in models:
                    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
                    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
                    results.append(cv_results)
                    names.append(name)
                    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    print(msg)

                return names, results

            ################################################################################
            #                                                                              #
            #                          Save Trained Model                                  #
            #                                                                              #
            ################################################################################
            def save_model(model, filename):
                pickle.dump(model, open(filename, 'wb'))

            ################################################################################
            #                                                                              #
            #                          Performance Measure                                 #
            #                                                                              #
            ################################################################################
            def classification_metrics(model, conf_matrix):
                print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
                print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
                ax.xaxis.set_label_position('top')
                plt.tight_layout()
                plt.title('Confusion Matrix', fontsize=20, y=1.1)
                plt.ylabel('Actual label', fontsize=15)
                plt.xlabel('Predicted label', fontsize=15)
                plt.show()
                print(classification_report(y_test, y_pred))

            # Load Dataset
            df = pd.read_csv('Data/crop_recommendation.csv')

            # Remove Outliers
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

            # Split Data to Training and Validation set
            target = 'label'
            X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

            # Train model
            pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
            crop_recommendation_model = pipeline.fit(X_train, y_train)
            y_pred = crop_recommendation_model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            classification_metrics(pipeline, conf_matrix)
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Plant Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            image = request.files.get('file')
            filename = image.filename
            file_path = os.path.join('static/images', filename)
            image.save(file_path)
            print(file_path)
            pred = prediction(file_path)
            print(pred)
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = file_path
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            return render_template('disease-result.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url,
                                   buy_link=supplement_buy_link)

        except:
            pass
    return render_template('disease.html', title=title)

@app.route('/crop_compare', methods=['GET', 'POST'])
def crop_compare():
    title = 'Crop Predicition'
    return render_template('cropRecommender.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
