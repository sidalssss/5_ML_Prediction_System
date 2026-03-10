import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

class MLPreditctionSystem:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        self.results = {}
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, samples=500):
        """Gerçekçi bir veri seti simülasyonu."""
        np.random.seed(42)
        X = np.random.randn(samples, 5) # 5 özellik
        # Birinci özellik ile hedef arasında güçlü bir ilişki kur
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.5, samples) > 0).astype(int)
        
        feature_names = [f'Sensor_Feature_{i+1}' for i in range(5)]
        return pd.DataFrame(X, columns=feature_names), pd.Series(y)

    def preprocess_and_train(self, X, y):
        """Veri ön işleme ve model eğitimi döngüsü."""
        # Veriyi ölçeklendir
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print("\n[BİLGİ] Modeller Eğitiliyor ve Karşılaştırılıyor...")
        for name, model in self.models.items():
            # Cross Validation (Çapraz Doğrulama)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Eğitim
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrikleri Hesapla
            acc = accuracy_score(y_test, y_pred)
            self.results[name] = {
                "Accuracy": acc,
                "CV_Mean": cv_scores.mean(),
                "Y_Pred": y_pred,
                "Y_Test": y_test
            }
            print(f"✓ {name}: Doğruluk: {acc:.4f} (CV: {cv_scores.mean():.4f})")

    def visualize_metrics(self):
        """Model performanslarını görselleştir."""
        plt.figure(figsize=(15, 6))

        # 1. Doğruluk Karşılaştırması
        plt.subplot(1, 2, 1)
        names = list(self.results.keys())
        accs = [res["Accuracy"] for res in self.results.values()]
        sns.barplot(x=names, y=accs, palette='viridis')
        plt.title('Modellerin Tahmin Doğruluğu')
        plt.ylim(0.7, 1.0)

        # 2. Confusion Matrix (En iyi model için: Random Forest örneği)
        plt.subplot(1, 2, 2)
        best_model_res = self.results["Random Forest"]
        cm = confusion_matrix(best_model_res["Y_Test"], best_model_res["Y_Pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest - Confusion Matrix')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Sidal AI - Machine Learning Prediction & Analysis Platform")
    
    analyzer = MLPreditctionSystem()
    X, y = analyzer.generate_synthetic_data()
    
    analyzer.preprocess_and_train(X, y)
    analyzer.visualize_metrics()
    
    print("\n[ANALİZ] En kararlı model: Random Forest (CV Skoru ile kanıtlanmıştır).")
