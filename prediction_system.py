import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Veri Seti Oluşturma (Örnek Data)
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Label': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

# 2. Ön İşleme
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modellerin Eğitilmesi
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# 4. Görselleştirme
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.ylabel('Doğruluk Oranı')
plt.title('Makine Öğrenmesi Modelleri Karşılaştırması')
plt.show()
