# Laporan Proyek Machine Learning - Endang Supriyadi

## Domain Proyek

Customer churn merupakan salah satu permasalahan paling krusial dalam industri layanan, termasuk telekomunikasi dan perbankan, karena kehilangan pelanggan berdampak langsung pada penurunan pendapatan, meningkatnya biaya akuisisi pelanggan baru, serta menurunnya performa bisnis secara keseluruhan. Berdasarkan penelitian terbaru, churn prediction dengan Machine Learning terbukti efektif untuk mengidentifikasi pelanggan yang berisiko churn sehingga perusahaan dapat melakukan tindakan preventif lebih awal [1].

Penelitian lain menunjukkan bahwa algoritma Machine Learning seperti Random Forest, Gradient Boosting, dan Neural Network mampu meningkatkan akurasi prediksi churn dan mendukung strategi retensi yang lebih terarah serta efisien [2]. Dalam konteks industri telekomunikasi, churn sering dipengaruhi oleh faktor seperti jenis kontrak, kualitas layanan, biaya langganan, serta pola konsumsi layanan pelanggan. Dengan memanfaatkan model prediksi, perusahaan dapat memahami pola perilaku pelanggan dan membuat keputusan berbasis data untuk meningkatkan loyalitas pelanggan.

Dengan demikian, proyek ini termasuk dalam domain Customer Analytics, khususnya fokus pada Customer Churn Prediction untuk perusahaan telekomunikasi. Model prediksi churn digunakan sebagai alat bantu pengambilan keputusan dalam strategi retensi, sehingga perusahaan dapat menurunkan tingkat churn dan meningkatkan profitabilitas.


## Business Understanding


### Problem Statements
Perusahaan telekomunikasi menghadapi tantangan besar dalam mempertahankan pelanggan lama karena tingginya tingkat churn. Ketika pelanggan berhenti berlangganan, perusahaan kehilangan pendapatan dan harus mengeluarkan biaya lebih besar untuk mendapatkan pelanggan baru.
Namun, proses identifikasi pelanggan yang berpotensi churn masih dilakukan secara manual atau berbasis asumsi sehingga tidak akurat dan tidak efektif.

Masalah bisnis yang ingin diselesaikan:

- Bagaimana perusahaan dapat memprediksi pelanggan mana yang berisiko churn sebelum mereka benar-benar berhenti?

- Faktor apa saja yang paling memengaruhi perilaku churn pelanggan?

- Bagaimana perusahaan dapat menggunakan prediksi ini untuk mengurangi churn dan meningkatkan retensi?

### Goals
Tujuan proyek ini adalah menghasilkan model Machine Learning yang dapat membantu perusahaan dalam pengambilan keputusan bisnis terkait retensi pelanggan.

🎯 Tujuan yang ingin dicapai:

- Membangun model prediksi churn yang mampu mengklasifikasikan pelanggan apakah berisiko churn atau tidak.

- Mengidentifikasi fitur yang paling berpengaruh terhadap keputusan pelanggan untuk churn.

- Menyediakan dasar rekomendasi strategi retensi yang dapat digunakan oleh perusahaan.

### Solution Statement

Permasalahan utama dalam proyek ini adalah tingginya tingkat customer churn, yaitu kondisi ketika pelanggan berhenti menggunakan layanan perusahaan. Untuk mengatasi hal tersebut, diperlukan sebuah sistem prediksi churn yang mampu mengidentifikasi pelanggan berisiko tinggi agar perusahaan dapat melakukan tindakan retensi dengan lebih cepat dan tepat sasaran.

Untuk menyelesaikan permasalahan ini, proyek ini membangun sebuah model Machine Learning berbasis Random Forest Classifier. Algoritma Random Forest dipilih sebagai solusi utama karena terbukti unggul dalam menangani data tabular, mampu mengatasi overfitting, dan memiliki kinerja baik pada dataset dengan fitur heterogen. Selain itu, Random Forest dapat memberikan interpretasi penting melalui feature importance, sehingga membantu perusahaan memahami faktor-faktor utama yang menyebabkan pelanggan melakukan churn.

Tahapan penyelesaian dilakukan melalui beberapa langkah berikut:

1. Data Preprocessing & Handling Imbalance
Dataset pelanggan diproses dengan pembersihan data, encoding fitur kategorikal, normalisasi, dan pembagian data.
Karena dataset bersifat imbalanced, teknik SMOTETomek digunakan untuk menyeimbangkan kelas sehingga model dapat belajar secara lebih adil pada kedua kelas (churn dan non-churn).

2. Pemodelan Menggunakan Random Forest Classifier
Model utama dibangun menggunakan algoritma Random Forest.
Dua skenario hyperparameter digunakan untuk melakukan model improvement:

    - Skenario 1 (Baseline Model): menggunakan parameter dasar Random Forest.

    - Skenario 2 (Tuned Model): menggunakan hyperparameter utama seperti n_estimators, max_depth, min_samples_split, min_samples_leaf, dan max_features.

    Perbandingan performa dilakukan untuk melihat peningkatan akurasi, presisi, recall, dan F1-score setelah tuning.

3. Evaluasi Model
Evaluasi dilakukan menggunakan:

    - Accuracy

    - Precision

    - Recall

    - F1-score

    - Confusion Matrix

    Metode evaluasi tersebut bertujuan memastikan model tidak hanya akurat tetapi juga mampu mendeteksi pelanggan churn dengan baik (recall tinggi).

4. Pemilihan Model Terbaik
Jika model tuning menunjukkan peningkatan signifikan terutama pada recall class “Churn”, maka model tersebut dipilih sebagai solusi final.
Model terbaik kemudian dijadikan dasar dalam memberikan wawasan faktor penyebab churn melalui analisis feature importance.

Dengan membangun dan mengoptimalkan model Random Forest ini, perusahaan dapat lebih proaktif dalam mengidentifikasi pelanggan berisiko churn dan menerapkan strategi retensi yang lebih efektif. Solusi ini mendukung pengambilan keputusan berbasis data dan dapat membantu perusahaan menurunkan tingkat churn secara signifikan di masa depan.

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

### Jumlah Dataset <br>

yaitu jumlah dataset mentah yang digunakan dalam membuat model ini.

Gambar 1: 

<img width="840" height="154" alt="image" src="https://github.com/user-attachments/assets/b53d7363-6211-4dd2-90d0-2d51efee7d5f" />

pada Gambar 1 dalam dataset ini terdapat 21 Column dan 7043 Rows


### Missing Value
Penting untuk melakukan pemeriksaan missing value karena akan sangat berpengaruh terhadap hasil model yang kita latih.

Gambar 2:

<img width="79" height="280" alt="image" src="https://github.com/user-attachments/assets/08609d35-e99a-4d38-90b8-4a8ec4753bf8" />

pada gambar 2 terdapat missing value pada column TotalCharges sebanyak 11

### Cek Jumlah Label Target <br>
ada ketidakseimbangan label target ini, tapi nantinya model ini akan berfokus pada target berlebel "yes" karena ingin memprediksi potensi customer melakukan churn

Gambar 3:

<img width="231" height="166" alt="image" src="https://github.com/user-attachments/assets/44ace6ad-19c8-488b-83fe-e92972fb3b02" />

pada Gambar 3 terdapat imbalanced terhadap label target. 

|Churn|Jumlah|
|-|-|
|No|5174|
|Yes|1869|


## Data Preparation
Melakukan transformasi data agar data bisa digunakan oleh model dengan melakukan proses cleaning, Encoding, On Hot Encoding 

### Menyederhanakan Label Kategori
Dalam column 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies' terdapat label kategori "No internet service" akan diubah menjadi Label Kategori "No". dan untuk column 'MultipleLines' terdapat label kategori "No phone service" akan diubah menjadi Label Kategori "No".

Hal ini dilakukan agar menyederhanakan pemprosessan data hasilnya dapat dilihat di Gambar 4.

Gambar 4

<img width="834" height="157" alt="image" src="https://github.com/user-attachments/assets/7d9c127f-c672-4a15-a33d-3f9ba58ccbea" />




### Menghapus Missing Value di Column TotalCharges
Data Cleaning ini bertujuan untuk membersihkan data agar data bisa diproses baik oleh model. Berikut hasil dari cleaning data:

Gambar 5 :

<img width="69" height="281" alt="image" src="https://github.com/user-attachments/assets/8e313a52-a052-4a33-8bc2-6791dc8e9901" />

pada Gambar 5 "TotalCharges" sudah 0, artinya data sudah bersih

### Menentukan Target
"Churn" sebagai Feature Target yang dimana bertype category/object jadi kita perlu melakukan Encoding dengan label 1 = "Yes", 0 ="No", selain itu kita melaukan drop pada CustomerID karena tidak digunakan

### Pemilihan Fitur
Hal ini dilakukan untuk mencari fitur mana yang mempengaruhi Churn
Gambar 6

<img width="502" height="189" alt="image" src="https://github.com/user-attachments/assets/4d1f74f0-7b52-48d0-b2cc-c9d05bbb4d95" />

Fitur yang dipilih adalah ['Contract', 'tenure', 'TotalCharges', 'MonthlyCharges', 'PaperlessBilling', 'OnlineSecurity', 'TechSupport', 'Dependents', 'SeniorCitizen', 'Partner', 'PaymentMethod']


### Memisahkan Fitur Numerik dan Category
tahapan ini agar mempermudah dalam pemrosesan, berikut pembagiannya : <br>
categorical_features = [
    'Contract', 'PaperlessBilling','OnlineSecurity', 'TechSupport', 'Dependents', 'SeniorCitizen', 'Partner', 'PaymentMethod', 'Churn'
]

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
<br>

### Melakukan On Hot Encoding (OHE) Pada column Kategori
hal ini agar program bisa membaca data kategori.

Gambar 7 :

<img width="848" height="81" alt="image" src="https://github.com/user-attachments/assets/8b434bb5-7627-4ee0-abac-23cd4784d946" />

Pada Gambar 7 data kategorikal membuat column baru dengan berisi nilai boolean karena menggunakan method 'get_dummies' mengubah string menjadi boolean seperti yang ada di gambar 8

Gambar 8 :

<img width="244" height="153" alt="image" src="https://github.com/user-attachments/assets/7c2cf203-85b4-4e6d-a555-d0e58ab5ddda" />

dalam case ini tidak menggunakan standarisasi pada column numerik karena model Random Forest ini tidak mementingkan standarisasi

### Spliting Data
dengan test_size = 20%, train_size = 80%, random_state=42, dan menggunakan statify = y. Hal ini dilakukan agar model tidak menghapal data <br>

**Stratify** digunakan untuk menjaga agar rasio kelas (proporsi target) dalam training set dan testing set sama dengan rasio kelas dalam dataset asli. y adalah variabel target.

### Melakukan Handle Imbalanced Data Target
Proses ini dilakukan hanya pada train, karena jika dilakukan pada data test juga data ini akan bocor dan bisa menyebabkan model overfitting atau underfitting. Tahap handle imbalance ini menggunakan method "SMOTETomek"

#### SMOTE (Synthetic Minority Oversampling Technique)
yaitu Menambah data kelas minoritas dengan membuat sampel sintetis (bukan sekadar duplikasi)

SMOTE bekerja dalam 3 langkah:
1. Cari tetangga terdekat,
Untuk setiap sampel minoritas, SMOTE mencari k-nearest neighbors dari kelas minoritas juga.

2. Pilih salah satu tetangga secara acak.

3. Buat sampel sintetis,
SMOTE menginterpolasi titik baru di antara dua sampel tadi.


Contoh ilustrasi sederhana:

$xnew = x_i + random(0,1) * (xneighbor - x_i)$


Artinya SMOTE membuat titik baru di antara dua titik asli.

➡️ Tujuan: menambah data minoritas sehingga seimbang dan distribusi tidak terduplikasi.


Hasil Handle Imbalance:

Gambar 9 :

<img width="139" height="57" alt="image" src="https://github.com/user-attachments/assets/0f6ee1b4-8829-4055-876e-b4d578ab2467" />


Pada Gambar 9 data train sudah balance dan siap untuk dilakukan training, proses ini hanya dilakukan di train.


## Modeling
Pengembangan model akan menggunakan algoritma machine learning yaitu Random Forest Clasifier dengan membedakan Hypertuning setiap modelnya.

### Model Random Forest
adalah algoritma ensemble yang terdiri dari banyak Decision Tree.
Setiap pohon membuat prediksi, lalu model mengambil suara terbanyak (majority vote) untuk menentukan kelas akhir. Karena menggabungkan banyak pohon, Random Forest lebih akurat, stabil, dan tidak mudah overfitting


#### Kelebihan

- Kuat terhadap overfitting (karena averaging banyak pohon)

- Akurasi tinggi untuk banyak kasus classification

- Mampu menangani data non-linear

- Tidak perlu scaling fitur

- Bisa menangani fitur kategori & numerik

- Robust terhadap missing value kecil

- Memiliki feature importance


#### Kekurangan
- Waktu training bisa lama (jika n_estimators besar)

- Model lebih sulit diinterpretasi dibanding Decision Tree

- Ukuran model lebih besar

- Jika dataset sangat besar, konsumsi memori tinggi


#### Parameter Utama Random Forest

1. n_estimators <br>
Jumlah pohon dalam forest.  <br>
Semakin banyak ➝ performa lebih stabil, waktu training lebih lama.

3. max_depth  <br>
Batas kedalaman pohon. <br>
Berguna untuk mencegah overfitting.

4. min_samples_split  <br>
Minimal jumlah sampel untuk memecah node.

5. min_samples_leaf  <br>
Minimal jumlah sampel pada leaf node.

6. max_features  <br>
Jumlah fitur yang digunakan ketika membagi node.

    - "sqrt" cocok untuk classification

    - "log2" alternatif

6. class_weight (jika imbalance)  <br>

    - "balanced" membuat model lebih adil pada kelas minoritas.

7. random_state  <br>
Untuk reproductibility.



### Skenario Model 1

  dengan hyper parameter :

| | **Hyperparamer** |  |
|-|----------|----------|
||n_estimators|200|
||min_samples_split|10|
||min_samples_leaf|5|
||max_features|'sqrt'|
||max_depth|12|
||random_state|42|
||class_weight|balanced|
||criterion|gini|


### Skenario Model 2
 dengan hyper parameter :

| | **Hyperparamer** |  |
|-|----------|----------|
||n_estimators|500|
||min_samples_split|5|
||min_samples_leaf|2|
||max_features|'sqrt'|
||max_depth|None|
||random_state|42|
||class_weight|balanced|
||criterion|entropy|


## Evaluasi
Membandingkan evaluasi setiap model mana yang lebih baik untuk dataset ini menggunakan precision, recall , accuracy, f1-score dan confusion matrix

Rumus Perhitungan Evaluasi model

- Accuracy = (TP + TN) / Total Data
Mengukur ketepatan keseluruhan model.

- Precision = TP / (TP + FP)
Mengukur seberapa tepat model saat memprediksi "Churn".

- Recall = TP / (TP + FN)
Mengukur kemampuan model menemukan pelanggan yang benar-benar churn (penting untuk kasus churn).

- F1-Score
= 2 × (Precision × Recall) / (Precision + Recall)
Digunakan ketika dataset tidak seimbang.


### Skenario Model 1

Gambar 10 

<img width="205" height="113" alt="image" src="https://github.com/user-attachments/assets/9254a455-862e-4318-a4d0-548ffe7eceb2" />

Penjelasan :
- accuracy : 0.7697228144989339

- Precision (0.55): Dari semua pelanggan yang diprediksi oleh model akan churn (Positif), hanya 55% yang benar-benar churn (True Positif). Ini berarti 45% dari prediksi churn model adalah salah (False Positive).

- Recall (0.70): Dari semua pelanggan yang sebenarnya churn (374 pelanggan), model berhasil mengidentifikasi 70% dari mereka (True Positif). Ini berarti 30% dari pelanggan churn terlewatkan (False Negative).

- F1-Score (0.62): Ini adalah rata-rata harmonik dari Precision dan Recall. F1-Score adalah metrik tunggal yang bagus untuk mengukur kinerja pada kelas minoritas.


Confusion Matrix :

- True Positive (TP = 260): Model dengan benar memprediksi 260 pelanggan akan churn. (Berhasil dicegah).

- False Negative (FN = 114): Model salah memprediksi 114 pelanggan tidak churn, padahal sebenarnya mereka churn. (Kesalahan Mahal: Kerugian pendapatan karena pelanggan churn tanpa peringatan).

- False Positive (FP = 210): Model salah memprediksi 210 pelanggan akan churn, padahal sebenarnya mereka tidak churn. (Biaya promosi atau insentif yang sia-sia).


### Skenario Model 2


Gambar 11 

<img width="191" height="113" alt="image" src="https://github.com/user-attachments/assets/6bbc87ac-6820-4b8e-bc2d-a502c6162699" />


Penjelasan :
- Accuracy: 0.7775408670931059

- Precision (Churn = 0.57): Dari semua yang model prediksi akan churn, 57% yang benar.

- Recall (Churn = 0.67): Dari semua pelanggan yang benar-benar churn, model berhasil menemukan 67% dari mereka.

- F1-Score (Churn = 0.62): Ukuran gabungan dari Precision dan Recall.


Confusion Matrix :
- True Positives (TP = 252): Jumlah pelanggan Churn yang berhasil diprediksi.

- False Negatives (FN = 122): Jumlah pelanggan Churn yang terlewatkan. (Kerugian).

- True Negatives (TN = 842): Jumlah pelanggan Non-Churn yang berhasil diprediksi.

- False Positives (FP = 191): Jumlah pelanggan Non-Churn yang salah diprediksi Churn. (Biaya).



**Kesimpulan** <br>
Rekomendasi Model Terbaik adalah Model 1.

Alasan Utama:

- Prioritas Recall yang Lebih Tinggi: Model 1 memiliki Recall 0.70 (versus 0.67 pada Model 2) dan False Negative (FN) yang lebih rendah (114). Dalam pencegahan churn, tujuannya adalah meminimalkan FN, karena setiap pelanggan churn yang terlewatkan (FN) berarti kerugian pendapatan yang langsung dan signifikan bagi perusahaan.

- Keseimbangan yang Dapat Diterima: Meskipun Model 2 memiliki Precision yang sedikit lebih tinggi (menghemat 19 kasus FP), peningkatan 8 kasus FN pada Model 2 kemungkinan besar lebih merugikan secara finansial daripada penghematan biaya insentif tersebut.

Oleh karena itu, Model 1 adalah model yang lebih baik untuk memitigasi risiko kerugian akibat churn karena lebih efektif dalam mengidentifikasi pelanggan yang benar-benar membutuhkan intervensi penyelamatan.


## Referensi Journal

[1] Al-Sultan, S.Y. and Al-Baltah, I.A., 2024. An improved random forest algorithm (ERFA) utilizing an unbalanced and balanced dataset to predict customer churn in the banking sector. IEEE Access.<br>
Link [https://ieeexplore.ieee.org/abstract/document/10516426]

[2] A. Manzoor, M. Atif Qureshi, E. Kidney and L. Longo, "A Review on Machine Learning Methods for Customer Churn Prediction and Recommendations for Business Practitioners," in IEEE Access, vol. 12, pp. 70434-70463, 2024, doi: 10.1109/ACCESS.2024.3402092. <br>
Link [https://ieeexplore.ieee.org/abstract/document/10531735]


