import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from config import DB_CONFIG

# Setup logging
logger = logging.getLogger("naive_bayes_logger")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MultinomialNBFromScratch:
    """
    Implementasi Multinomial Naive Bayes dari awal tanpa scikit-learn.
    """
    def __init__(self, alpha=1.0):
        # alpha adalah parameter untuk Laplace Smoothing.
        self.alpha = alpha

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Inisialisasi parameter yang akan dipelajari model.
        self.class_priors_ = np.zeros(n_classes)
        self.feature_counts_ = np.zeros((n_classes, n_features))
        self.class_counts_ = np.zeros(n_classes)

        # Iterasi untuk menghitung parameter dari data training.
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_counts_[i] = X_c.shape[0]
            self.class_priors_[i] = np.log(self.class_counts_[i] / n_samples) #prior
            self.feature_counts_[i, :] = X_c.sum(axis=0) + self.alpha

        # Menghitung probabilitas logaritmik untuk setiap fitur per kelas.
        total_words_per_class = self.feature_counts_.sum(axis=1)
        self.feature_log_prob_ = np.log(self.feature_counts_ / total_words_per_class[:, np.newaxis]) #likelihood
        
        return self

    def _predict_log_proba(self, X):
        # Menghitung probabilitas logaritmik posterior untuk prediksi.
        return X @ self.feature_log_prob_.T + self.class_priors_  #posterior

    def predict(self, X):
        # Melakukan prediksi label kelas.
        log_probas = self._predict_log_proba(X)
        return self.classes_[np.argmax(log_probas, axis=1)]


def save_results_to_db(original_texts, actual_labels, predicted_labels):
    """
    Menyimpan hasil klasifikasi ke database.
    """
    if len(original_texts) == 0:
        logger.warning("Tidak ada hasil prediksi untuk disimpan.")
        return
    try:
        logger.info("Menyimpan hasil klasifikasi ke database...")
        results_df = pd.DataFrame({
            'teks': original_texts,
            'sentimen_aktual': actual_labels,
            'sentimen_prediksi': predicted_labels
        })
        db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_url)
        results_df.to_sql(name='hasil_klasifikasi', con=engine, if_exists='replace', index=False)
        logger.info(f"Berhasil menyimpan {len(results_df)} baris hasil prediksi.")
    except Exception as e:
        logger.error(f"Gagal menyimpan hasil klasifikasi: {e}", exc_info=True)


def train_and_evaluate_nb(feature_extraction_result):
    """
    Fungsi utama untuk melatih dan mengevaluasi model Naive Bayes,
    serta menghasilkan log visualisasi dan hasil prediksi rinci.
    """
    try:
        logger.info("Memulai pelatihan dan evaluasi Naive Bayes...")

        X_train = feature_extraction_result.get('X_train_tfidf')
        y_train = feature_extraction_result.get('y_train')
        X_test_tfidf = feature_extraction_result.get('X_test_tfidf')
        y_test = feature_extraction_result.get('y_test')
        X_test_text = feature_extraction_result.get('X_test_text', [])
        vectorizer = feature_extraction_result.get('vectorizer')

        if X_train is None or y_train is None or X_train.shape[0] == 0:
            return {'success': False, 'error': 'Data training tidak lengkap atau kosong.'}

        logger.info("Melatih model MultinomialNB (versi dari awal)...")
        model = MultinomialNBFromScratch(alpha=1.0)
        model.fit(X_train, y_train)
        logger.info("Pelatihan model selesai.")

        y_pred = model.predict(X_test_tfidf)
        save_results_to_db(X_test_text, y_test, y_pred)

        logger.info("Menghitung metrik evaluasi...")
        accuracy = accuracy_score(y_test, y_pred)
        labels = sorted(list(set(y_test) | set(y_pred)))
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        logger.info(f"Akurasi: {accuracy:.4f}")
        
        training_history = []
        training_history.append({'type': 'header', 'text': 'Detail Proses Pelatihan Naive Bayes'})
        training_history.append({'type': 'step', 'text': 'Langkah 1: Menghitung Jumlah Dokumen per Kelas'})
        for i, c in enumerate(model.classes_):
            training_history.append({'type': 'log', 'text': f"- Kelas '{c}': {int(model.class_counts_[i])} dokumen"})
        training_history.append({'type': 'step', 'text': 'Langkah 2: Menghitung Probabilitas Prior Kelas'})
        for i, c in enumerate(model.classes_):
            prior = np.exp(model.class_priors_[i])
            training_history.append({'type': 'log', 'text': f"- P({c}) = {prior:.4f}"})
        if vectorizer:
            feature_names = vectorizer.get_feature_names_out()
            training_history.append({'type': 'step', 'text': 'Langkah 3: Menghitung Probabilitas Fitur (Likelihood)'})
            for i, c in enumerate(model.classes_):
                training_history.append({'type': 'log', 'text': f"  5 Fitur Teratas untuk Kelas '{c}':"})
                top_5_indices = np.argsort(model.feature_log_prob_[i])[-5:]
                for feature_index in reversed(top_5_indices):
                    word = feature_names[feature_index]
                    prob = np.exp(model.feature_log_prob_[i][feature_index])
                    training_history.append({'type': 'log', 'text': f"    - P('{word}'|{c}) ≈ {prob:.6f}"})
        
        # --- PERUBAHAN BARU: Membuat daftar hasil prediksi rinci ---
        prediction_results = []
        # Memastikan y_test adalah list atau bisa diakses dengan .iloc
        y_test_list = y_test.tolist() if isinstance(y_test, pd.Series) else y_test
        for i in range(len(X_test_text)):
            prediction_results.append({
                'text': X_test_text[i],
                'actual': y_test_list[i],
                'predicted': y_pred[i]
            })
        # --- AKHIR PERUBAHAN ---

        return {
            'success': True,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'training_history': training_history,
            'prediction_results': prediction_results # Menambahkan hasil prediksi ke dictionary
        }

    except Exception as e:
        logger.error(f"Error saat melatih atau mengevaluasi model: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
