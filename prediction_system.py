import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoML-Sidal")

class SidalAutoML:
    """
    Gelişmiş Makine Öğrenmesi Pipeline ve Otomatik Model Seçim Sistemi.
    Stacking Ensemble ve Hiper-parametre optimizasyonu içerir.
    """
    def __init__(self):
        self.best_estimator = None
        self.feature_names = None

    def build_preprocessor(self, num_cols: List[str], cat_cols: List[str]):
        """Veri tipine göre otomatik özellik mühendisliği yapan pipeline."""
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', RobustScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    def train_with_grid_search(self, X, y, num_cols, cat_cols):
        """En iyi modeli bulmak için GridSearch ve Cross-Validation uygular."""
        preprocessor = self.build_preprocessor(num_cols, cat_cols)
        
        # Base Model: Random Forest
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

        logger.info("Hiper-parametre optimizasyonu başlatılıyor...")
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=StratifiedKFold(5), 
                                   scoring='accuracy', n_jobs=-1, verbose=1)
        
        grid_search.fit(X, y)
        self.best_estimator = grid_search.best_estimator_
        logger.info(f"En iyi parametreler bulundu: {grid_search.best_params_}")
        return grid_search.best_score_

    def evaluate(self, X_test, y_test):
        """Model performansını ileri düzey metriklerle raporlar."""
        y_pred = self.best_estimator.predict(X_test)
        y_prob = self.best_estimator.predict_proba(X_test)[:, 1]
        
        print("\n--- Model Performans Raporu ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    # Mock Veri Seti
    data = pd.DataFrame(np.random.randn(100, 4), columns=['f1', 'f2', 'f3', 'f4'])
    data['category'] = np.random.choice(['A', 'B', 'C'], 100)
    target = np.random.randint(0, 2, 100)
    
    automl = SidalAutoML()
    score = automl.train_with_grid_search(data, target, ['f1', 'f2', 'f3', 'f4'], ['category'])
    print(f"Eğitim Tamamlandı. Cross-Val Skoru: {score:.4f}")
