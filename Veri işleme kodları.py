import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri ön işleme
veri = pd.read_csv('C:/Users/OneDrive/Desktop/Belgeler/eksikveriler.csv')
print("Ham Veri:\n", veri)

# Boy sütununu ayırma
boy = veri[['boy']]
print("\nBoy Sütunu:\n", boy)

# Eksik veri işleme
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Yas = veri.iloc[:, 1:4].values  # Yaş, boy, kilo sütunları
print("\nYaş, Boy, Kilo (Eksik Verilerle):\n", Yas)

# Eksik verileri ortalama ile doldurma
imputer.fit(Yas)  # Tüm sütunlar için fit işlemi
Yas = imputer.transform(Yas)
print("\nYaş, Boy, Kilo (Eksik Veriler Dolduruldu):\n", Yas)

# Ülke verilerini ayırma
ulke = veri.iloc[:, 0:1].values
print("\nÜlke Verileri:\n", ulke)

# Label Encoding (Kategorik > Numerik)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veri.iloc[:, 0])
print("\nLabel Encoding Sonrası Ülke Verileri:\n", ulke)

# One-Hot Encoding
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print("\nOne-Hot Encoding Sonrası Ülke Verileri:\n", ulke)

# DataFrame'lere dönüştürme
sonuc = pd.DataFrame(data=ulke, index=range(len(veri)), columns=['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data=Yas, index=range(len(veri)), columns=['boy', 'kilo', 'yas'])
print("\nOne-Hot Encoding Sonucu:\n", sonuc)
print("\nYaş, Boy, Kilo DataFrame:\n", sonuc2)

# Cinsiyet verilerini ayırma
cinsiyet = veri.iloc[:, -1].values
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(len(veri)), columns=['cinsiyet'])
print("\nCinsiyet Verileri:\n", sonuc3)

# DataFrame'leri birleştirme
s = pd.concat([sonuc, sonuc2], axis=1)
print("\nBirleştirilmiş DataFrame:\n", s)

# Veriyi eğitim ve test setlerine ayırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0, stratify=sonuc3)
print("\nEğitim Seti Boyutu:", x_train.shape)
print("Test Seti Boyutu:", x_test.shape)

# Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
print("\nÖlçeklendirilmiş Eğitim Verisi (İlk 5 Satır):\n", X_train[:5])
print("\nÖlçeklendirilmiş Test Verisi (İlk 5 Satır):\n", X_test[:5])