import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DB_CONFIG
import logging
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_split_data_from_db(engine):
    """
    Memuat data yang sudah dibagi (termasuk label) langsung dari tabel 'pembagian_data'.
    """
    try:
        logger.info("Membaca data latih/uji langsung dari 'pembagian_data'...")
        query = """
        SELECT 
            teks AS stemming, 
            sentiment_pakar, 
            data_type
        FROM 
            pembagian_data
        WHERE 
            sentiment_pakar IS NOT NULL AND sentiment_pakar != '';
        """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning("Tidak ada data berlabel yang ditemukan di 'pembagian_data'.")
            return None, None

        train_df = df[df['data_type'] == 'training'].copy()
        test_df = df[df['data_type'] == 'testing'].copy()
        
        logger.info(f"Berhasil memuat data: {len(train_df)} baris training, {len(test_df)} baris testing.")
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Gagal membaca data dari 'pembagian_data': {e}", exc_info=True)
        return None, None

def log_extraction_details(vectorizer, tfidf_matrix, raw_texts):
    """Mencetak detail proses ekstraksi fitur."""
    feature_names = vectorizer.get_feature_names_out()
    sample_indices = [0, 1] # Ambil dua sampel pertama

    logger.info("\n" + "="*20 + " DETAIL EKSTRAKSI FITUR " + "="*20)
    for sample_index in sample_indices:
        if len(raw_texts) <= sample_index:
            continue

        sample_text = raw_texts.iloc[sample_index]
        sample_matrix_row = tfidf_matrix[sample_index]

        logger.info(f"\n[Sampel Teks]: '{sample_text}'")
        feature_index = sample_matrix_row.nonzero()[1]
        tfidf_scores = zip(feature_index, [sample_matrix_row[0, x] for x in feature_index])
        
        feature_scores = {feature_names[j]: score for j, score in tfidf_scores}
        sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5] # Tampilkan 5 teratas
        
        logger.info("  Top 5 Fitur TF-IDF:")
        for feature, score in sorted_scores:
            logger.info(f"    - '{feature}': {score:.4f}")

def extract_and_transform_features(train_df, test_df, use_smote=True):
    """Melakukan ekstraksi fitur TF-IDF dan secara opsional menjalankan SMOTE."""
    if train_df.empty or test_df.empty:
        logger.warning("Data training atau testing kosong.")
        return None

    X_train = train_df['stemming']
    y_train = train_df['sentiment_pakar']
    X_test = test_df['stemming']
    y_test = test_df['sentiment_pakar']
    
    train_count_before_smote = len(y_train)
    distribution_before = Counter(y_train)

    logger.info("Menginisialisasi TF-IDF Vectorizer dengan n-gram (1, 2)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    logger.info(f"Ekstraksi fitur TF-IDF selesai. Jumlah fitur total: {len(tfidf_vectorizer.get_feature_names_out())}")
    log_extraction_details(tfidf_vectorizer, X_train_tfidf, X_train)
    
    # --- PERUBAHAN: Logika SMOTE dibuat kondisional ---
    if use_smote:
        logger.info("\n" + "="*20 + " PROSES PENYEIMBANGAN DATA (SMOTE) " + "="*20)
        logger.info(f"Distribusi kelas sebelum SMOTE: {distribution_before}")

        min_class_count = min(distribution_before.values()) if distribution_before else 1
        k_neighbors_safe = max(1, min_class_count - 1)
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_safe)
        logger.info(f"SMOTE diinisialisasi dengan k_neighbors={k_neighbors_safe}")
        
        X_resampled_tfidf, y_resampled = smote.fit_resample(X_train_tfidf, y_train)
        
        distribution_after = Counter(y_resampled)
        logger.info(f"Distribusi kelas setelah SMOTE: {distribution_after}")
    else:
        logger.info("\n" + "="*20 + " PROSES PENYEIMBANGAN DATA (DILEWATI) " + "="*20)
        logger.info("SMOTE tidak diaktifkan.")
        X_resampled_tfidf = X_train_tfidf
        y_resampled = y_train
        distribution_after = distribution_before # Distribusi tidak berubah
    # --- AKHIR PERUBAHAN ---

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    logger.info("Vectorizer TF-IDF berhasil disimpan.")

    return {
        "X_train_tfidf": X_resampled_tfidf,
        "y_train": y_resampled,
        "X_test_tfidf": X_test_tfidf,
        "y_test": y_test,
        "vectorizer": tfidf_vectorizer,
        "X_test_text": X_test.tolist(),
        "train_count_before_smote": train_count_before_smote,
        "distribution_before": dict(distribution_before),
        "distribution_after": dict(distribution_after)
    }

def main_feature_extraction_pipeline(use_smote=True):
    """Pipeline utama untuk ekstraksi fitur, menerima parameter use_smote."""
    logger.info("="*50)
    logger.info(f"MEMULAI PIPELINE EKSTRAKSI FITUR (use_smote={use_smote})")
    logger.info("="*50)
    
    try:
        db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_url)

        train_df, test_df = load_split_data_from_db(engine)
        if train_df is None or test_df is None:
            return {'success': False, 'error': 'Gagal memuat data. Jalankan pembagian data terlebih dahulu.'}

        feature_results = extract_and_transform_features(train_df, test_df, use_smote=use_smote)
        if not feature_results:
            return {'success': False, 'error': 'Gagal melakukan ekstraksi fitur.'}

        train_count_after = feature_results['X_train_tfidf'].shape[0]
        train_count_before = feature_results['train_count_before_smote']
        smote_added_count = train_count_after - train_count_before
        
        stats = {
            'total_data': len(train_df) + len(test_df),
            'train_data_count_before': train_count_before,
            'train_data_count_after': train_count_after,
            'test_data_count': len(test_df),
            'smote_added_count': smote_added_count,
            'feature_count': len(feature_results['vectorizer'].get_feature_names_out()),
            'distribution_before': feature_results['distribution_before'],
            'distribution_after': feature_results['distribution_after']
        }
        
        logger.info("Pipeline ekstraksi fitur selesai.")
        
        return {
            'success': True,
            'stats': stats,
            **feature_results
        }

    except Exception as e:
        logger.error(f"Error fatal dalam pipeline ekstraksi fitur: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    main_feature_extraction_pipeline(use_smote=True)
