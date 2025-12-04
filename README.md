# Laporan Proyek Machine Learning - Endang Supriyadi

## Domain Proyek

Dalam sebuah kegiatan usaha, perusahaan atau pelaku usaha akan mencari cara agar mempertahankan customernya, melihat persaingan yang ketat dengan kompetitor saling berjuang untuk mempertahankan bahkan meningkat penjualan, hal itu bisa terjadi potensi churn karena customer akan membandingkan dengan kompetitor kita[1].

Churn dapat terjadi karena berbagai alasan seperti kualitas layanan dan respon terhadap kenaikan harga [2]. Maka dari itu sangat penting untuk melakukan potensi tipe pelanggan yang berpeluang untuk melakukan churn dengan menggunakan machine learning agar kita bisa memprediksi customer churn secara otomatis dan cepat, karena semakin cepat diketahui maka perusahaan bisa mengambil langkah agar customer tidak melakukan churn.


## Business Understanding


### Problem Statements

- Dari fitur yang ada fitur mana yang memiliki pengaruh besar pada churn?
- Model machine learning mana antara BalancedRandomForestClassifier dan XGBOOST yang bagus untuk memprediksi potensi churn

### Goals
- Mengetahui fitur mana yang paling berpengaruh terhadap potensi churn.
- menemukan model yang bagus untuk memprediksi potensi churn

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.


## Data Understanding
Dataset ini menggunakan dataset Telco Customer link. Contoh: [Telco Customer](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download).


### Variabel-variabel pada Telco Customer dataset adalah sebagai berikut:
- **CustomerID**: A unique ID that identifies each customer. <br>
- **Gender**: The customer’s gender: Male, Female <br>
- **Senior Citizen**: Indicates if the customer is 65 or older: Yes, No <br>
- **Partner** : Whether the customer has a partner or not (Yes, No) <br>
- **Dependents** : Whether the customer has dependents or not (Yes, No) <br>
- **Tenure** : Number of months the customer has stayed with the company <br>
- **PhoneService** : Whether the customer has a phone service or not (Yes, No) <br> 
- **MultipleLines** : Whether the customer has multiple lines or not (Yes, No, No phone service) <br>
- **InternetService** : Customer’s internet service provider (DSL, Fiber optic, No) <br>
- **OnlineSecurity** : Whether the customer has online security or not (Yes, No, No internet service) <br>
- **Online Backup**: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No <br>
- **Device Protection**: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No <br>
- **Tech Support**: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No <br>
- **Streaming TV**: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service. <br>
- **Streaming Movies**: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service. <br>
- **Contract**: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year. <br>
- **Paperless Billing**: Indicates if the customer has chosen paperless billing: Yes, No <br>
- **Payment Method**: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check <br>
- **Monthly Charge**: Indicates the customer’s current total monthly charge for all their services from the company. <br>
- **Total Charges**: Indicates the customer’s total charges, calculated to the end of the quarter specified above. <br>
- **Churn**: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.

**Type Data dalam dataset** <br>
Ada 21 Column dalam dataset

Gambar 1: 

<img src="https://i.ibb.co/RGGvMfRf/Screenshot-2025-11-25-174212.png" alt="Screenshot-2025-11-25-174212" border="0">

pada Gambar 1 type data object itu column berkategori, hanya ada 3 type numerik sisanya category

Gambar 2:

<img src="https://i.ibb.co/Mxp0PqFp/Screenshot-2025-11-25-224539.png" alt="Screenshot-2025-11-25-224539" border="0">

pada gambar 2 bahwa dalam dataset ada 21 column dan 7043 rows

**Cek Jumlah Label Target** <br>
ada ketidakseimbangan label target ini, tapi nantinya model ini akan berfokus pada target berlebel "yes" karena ingin memprediksi potensi customer melakukan churn

Gambar 3:

<img src="https://i.ibb.co/TBPt1qH2/download.png" alt="download" border="0">

pada Gambar 3 terdapat imbalanced terhadap label target. Perlu jadi perhatian penting terhadap data imbalance ini.


## Data Preparation
Melakukan transformasi data agar data bisa digunakan oleh model dengan melakukan proses cleaning, Encoding, On Hot Encoding 

### Mencari Kemungkinan Nilai Kosong
dataset ini cukup bersih dan tidak ditemukan data yang kotor tapi ternyata data ini mengandung nilai kosong, sehingga saya mengatasinya dengan regex mengganti nilai kosong dengan nan dan ditemukanlah pada "TotalCharges" ada 11 nilai kosong.

Gambar 4 :

<img src="https://i.ibb.co.com/8LwcfKSf/Screenshot-2025-11-25-183233.png" alt="Screenshot-2025-11-25-183233" border="0">

untuk itu perlu dilakukan cleaning data "TotalCharges"

### Data Cleaning
Data Cleaning ini bertujuan untuk membersihkan data agar data bisa diproses baik oleh model. Berikut hasil dari cleaning data:

Gambar 5 :

<img src="https://i.ibb.co.com/6Vk1G1n/Screenshot-2025-11-25-183514.png" alt="Screenshot-2025-11-25-183514" border="0">

pada Gambar 5 "TotalCharges" sudah 0, artinya data sudah bersih

### Menentukan Target
"Churn" sebagai Feature Target yang dimana bertype category/object jadi kita perlu melakukan Encoding dengan label 1 = "Yes", 0 ="No", selain itu kita melaukan drop pada CustomerID karena tidak digunakan

### Memisahkan Fitur Numerik dan Category
tahapan ini agar mempermudah dalam pemrosesan, berikut pembagiannya : <br>
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
<br>

### Spliting Data
dengan test_size = 20%, train_size = 80%, random_state=42, dan menggunakan statify = y. <br>

**Stratify** digunakan untuk menjaga agar rasio kelas (proporsi target) dalam training set dan testing set sama dengan rasio kelas dalam dataset asli. y adalah variabel target.

### Melakukan ColumnTransformer
Fungsi utamanya adalah untuk memastikan bahwa fitur kategorikal di-encode menjadi format numerik yang dapat dipahami oleh model (Machine Learning), sementara fitur numerik lainnya diteruskan tanpa diubah.

Dalam proses ini dilakukan On Hot Encode untuk data categorical, dengan *handle_unknown='ignore'*: Menginstruksikan encoder untuk mengabaikan (mengubahnya menjadi semua $0$) kategori baru yang muncul di data testing tetapi tidak ada di data training. Ini mencegah error saat deployment.

*sparse_output=False*: Memastikan output yang dihasilkan adalah dense array (penuh dengan angka $0$ dan $1$) daripada sparse matrix (yang lebih efisien memori tetapi lebih sulit ditangani di awal).

### Analysis Feature yang berpengaruh terjadinya Churn

ini gambaran visualisasi korelasi <br>

Gambar 6 :
<img src="https://i.ibb.co/7NvF4nVd/download-1.png" alt="korelasi" border="0">

Dari Gambar 6 didapatkan bahwa contrak month to month berperan sangat besar terhadap terjadinya customer churn


## Modeling
Tahap ini dimana tahap pembuatan model menjadi tantangan besar karena data label imbalance maka dari itu saya mencoba perbandingan model BalancedRandomForestClassifier dan XGBOOST karena model ini bisa menghandle masalah imbalance
pada setiap model menggunakan threshold=0.7

- Model BalancedRandomForestClassifier

  dengan hyper parameter :

| | **Hyperparamer** |  |
|-|----------|----------|
||n_estimators|200|
||min_samples_split|5|
||min_samples_leaf|5|
||max_features|'sqrt'|
||max_depth|20|
||random_state|42|
||class_weight|balanced|


- Model XGBOOST 
 dengan hyper parameter :

| | **Hyperparamer** |  |
|-|----------|----------|
||n_estimators|500|
||learning_rate|0.05|
||subsample|0.8|
||colsample_bytree|0.8|
||max_depth|6|
||random_state|42|
||scale_pos_weight|2.5|
||eval_metric|"logloss"|


tahap ini menggunakan pipeline agar nantinya ketika ingin mencoba model dengan data baru tidak usah melakukan encode atau OHE secara manual. 

## Evaluasi
Membandingkan evaluasi setiap model mana yang lebih baik untuk dataset ini menggunakan precision, recall , accuracy dan f1-score

Gambar 7

<img src="https://i.ibb.co/Cs0ZN2RF/Screenshot-2025-11-25-211453.png" alt="Screenshot-2025-11-25-211453" border="0">

**Perbandingan** <br>
**Untuk model balanced random forest**
dari output diatas nilai akurasi 0.77, dengan fokus utama kelas 1 melihat dari hasil evaluasi bahwa model ini mendapatkan :
- Nilai recall = 0.75 yakni mendeteksi sebanyak 280 customer(True Positive) dari 374 dan ada 94 customer yang dilewatkan
- Nilai precission = 0.55 Dari semua customer yang diprediksi model akan churn (280 TP + 231 FP), hanya 55% prediksi yang benar. Model ini memiliki 231 False Positives (FP).
- F1-score = 0.63, Ini adalah rata-rata harmonis antara Precision dan Recall. Nilai 0.63menunjukkan model yang cukup seimbang.

**Untuk model XGBOOST**
dari output diatas nilai akurasi 0.78, dengan fokus utama kelas 1 melihat dari hasil evaluasi bahwa model ini mendapatkan :
- Nilai recall = 0.50 yakni mendeteksi sebanyak 187 customer(True Positive) dari 374 dan ada 187 customer yang dilewatkan
- Nilai precission = 0.59 Dari semua customer yang diprediksi model akan churn (187 TP + 128 FP), hanya 59% prediksi yang benar. Model ini memiliki 128 False Positives (FP).
- F1-score = 0.54, Ini adalah rata-rata harmonis antara Precision dan Recall. Nilai 0.54menunjukkan model yang cukup seimbang.

**Kesimpulan** <br>
Model Balanced Random Forest lebih bagus dari model XGBOOST. Meskipun XGBoost memiliki Akurasi dan Precision sedikit lebih tinggi, Recall 0.75 dari BRF secara langsung menerjemahkan menjadi risiko bisnis yang jauh lebih rendah (kehilangan 94 customer) dibandingkan risiko kerugian yang tinggi dari XGBoost (kehilangan 187 customer). BRF adalah model yang lebih praktis dan cost-effective untuk tugas pencegahan churn.


## Referensi Journal

[1] Al-Sultan, S.Y. and Al-Baltah, I.A., 2024. An improved random forest algorithm (ERFA) utilizing an unbalanced and balanced dataset to predict customer churn in the banking sector. IEEE Access.<br>
Link [https://ieeexplore.ieee.org/abstract/document/10516426]

[2] A. Manzoor, M. Atif Qureshi, E. Kidney and L. Longo, "A Review on Machine Learning Methods for Customer Churn Prediction and Recommendations for Business Practitioners," in IEEE Access, vol. 12, pp. 70434-70463, 2024, doi: 10.1109/ACCESS.2024.3402092. <br>
Link [https://ieeexplore.ieee.org/abstract/document/10531735]


