import streamlit as st
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Sequential  # used to initialize our ANN
from tensorflow.keras.layers import Dense  # used to make layers in ANN
from tensorflow.keras.layers import Dropout


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def loadData():
    dataset = pd.read_csv('cleveland.csv')
    return dataset

# Basic preprocessing required for all the models.


def preprocessing(dataset):
    dataset.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                       'fbs', 'restecg', 'thalach', 'exang',
                       'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataset['target'] = dataset.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    dataset['thal'] = dataset.thal.fillna(dataset.thal.mean())
    dataset['ca'] = dataset.ca.fillna(dataset.ca.mean())

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Training Neural Network for Classification.


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def neuralNet(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # train data
    # initializing the ANN
    classifier = Sequential()

    # add input layer and first hidden layer
    classifier.add(Dense(activation="relu", input_dim=13,
                   units=7, kernel_initializer="uniform"))
    # classifier.add(Dropout(0.3))
    # add second hidden layer
    classifier.add(Dense(activation="relu", units=7,
                   kernel_initializer="uniform"))
    # classifier.add(Dropout(0.3))
    # add output layer
    classifier.add(Dense(activation="sigmoid", units=1,
                   kernel_initializer="uniform"))

    # Compiling the ANN
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to the Training set
    classifier.fit(X_train, y_train, batch_size=32, epochs=500)
    # Scalling the data before feeding it to the Neural Network.
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    score1 = accuracy_score(y_test, y_pred)*100
    report = classification_report(y_test, y_pred)

    return score1, report, classifier


def prediction(name, age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)
    score1, report, classifier = neuralNet(X_train, X_test, y_train, y_test)
    prediksi = classifier.predict(sc.fit_transform(np.array(
        [[age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal]])))
    prediksi_perc = int(prediksi * 100)
    st.write("Saudara ", name,
             " Memiliki kemungkinan memiliki penyakit jantung sebesar ", prediksi_perc, "%")


def main():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    st.title("PREDIKSI PENYAKIT JANTUNG")
    st.markdown(
        "Aplikasi ini dibuat untuk memprediksi kemungkinan sesorang mengidap penyakit jantung")
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)
    score1, report, classifier = neuralNet(X_train, X_test, y_train, y_test)
    st.write("Tingkat Keakuratan Model : ", score1)
    name = st.text_input("Masukkan nama anda : ", 'nama')
    age = st.number_input("Masukkan usia anda", 0, 100, 0)
    pilih_sex = st.radio("Pilih jenis kelamin ", options=[
                         'Laki-laki', 'Perempuan'])
    sex = 0
    if pilih_sex == 'Laki-laki':
        sex = 1
    else:
        sex = 0
    display_cp = st.checkbox(
        "Apakah anda mengalami sakit di dada ? *centang jika iya")
    cp = 0
    if display_cp:
        pilih_cp = st.radio("Pilih jenis sakit yang anda rasakan : ", options=[
                            'Typical Angina', 'Atypical angina', 'Non-anginal pain', 'Asymptotic'])
        if pilih_cp == 'Typical Angina':
            cp = 1
        elif pilih_cp == 'Atypical angina':
            cp = 2
        elif pilih_cp == 'Non-anginal pain':
            cp = 3
        else:
            cp = 4
    tresthbp = st.number_input(
        "Masukkan tekanan darah anda (/mmHg) : ", 50, 400, 120)
    chol = st.number_input(
        "Masukkan kadar kolesterol darah anda (mg/dL) :", 100, 400, 180)
    pilih_fbs = st.checkbox(
        "Apakah anda memiliki gula daran lebih dari 120mg/dL ? Centang jika iya")
    fbs = 0
    if pilih_fbs:
        fbs = 1
    pilih_restecg = st.radio("Pilih hasil elektrokardiografi anda : ", options=[
                             'Normal', 'ST-T wave abnormality', 'Left ventricular hyperthroph'])
    restecg = 0
    if pilih_restecg == 'Normal':
        restecg = 0
    elif pilih_restecg == 'ST-T wave abnormality':
        restecg = 1
    else:
        restecg = 2
    thalach = st.number_input(
        "Masukkan tekanan darah tertinggi anda (/mmHg) : ", 50, 400, 120)
    pilih_exang = st.checkbox(
        "Apakah ketika melakukan aktivitas fisik berat seperti olahraga , dada anda terasa sakit ? Centang jika iya")
    exang = 0
    if pilih_exang:
        exang = 1
    oldpeak = st.number_input("Masukkan nilai ST Depression : ", 0, 10, 1)
    pilih_slope = st.radio("Masukkan bentuk kurva ST ", options=[
                           'Unsloping', 'Flat', 'Downsloping'])
    slope = 1
    if pilih_slope == 'Unsloping':
        slope = 1
    elif pilih_slope == 'Flat':
        slope = 2
    else:
        slope = 3
    cs = st.number_input(
        "Masukkan pembuluh darah utama diwarnai dengan fluoroskopi", 0, 3, 1)
    pilih_thal = st.radio("Kelainan Thalassemia ", options=[
                          'Normal', 'Fixed defect', 'reversible defect'])
    thal = 3
    if pilih_thal == 'Normal':
        thal = 3
    elif pilih_thal == 'Fixed defect':
        thal = 6
    else:
        thal = 7
    st.write(name, age, sex, cp, tresthbp, chol, fbs, restecg,
             thalach, exang, oldpeak, slope, cs, thal)
    butt = st.button('Prediksi')
    if butt:
        prediction(name, age, sex, cp, tresthbp, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, cs, thal)


if __name__ == "__main__":
    main()
