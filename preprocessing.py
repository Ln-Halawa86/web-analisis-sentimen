# --- Mengimpor Pustaka (Library) yang Diperlukan ---
import pandas as pd  # Digunakan untuk manipulasi data, terutama dalam bentuk DataFrame.
from sqlalchemy import create_engine  # Digunakan untuk membuat koneksi ke database SQL.
from config import DB_CONFIG  # Mengimpor konfigurasi database (host, user, password, dll) dari file terpisah.
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Mengimpor class untuk membuat stemmer (mengubah kata ke bentuk dasar).
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Mengimpor class untuk membuat penghapus stopword.
import re  # Pustaka untuk Regular Expressions, digunakan untuk membersihkan teks dengan pola tertentu.
import logging  # Pustaka untuk mencatat (logging) informasi, peringatan, dan error selama program berjalan.
import time  # Pustaka untuk mengukur waktu eksekusi program.

# --- Konfigurasi Logging ---
# Mengatur konfigurasi dasar untuk logging agar menampilkan waktu, level log, dan pesan.
logging.basicConfig(
    level=logging.INFO,  # Menetapkan level minimum log yang akan ditampilkan (INFO, WARNING, ERROR).
    format='%(asctime)s - %(levelname)s - %(message)s'  # Menetapkan format pesan log.
)
# Membuat objek logger utama yang akan digunakan di seluruh skrip.
logger = logging.getLogger(__name__)

# --- Inisialisasi Sastrawi & Engine Database ---
try:
    # Memberi informasi bahwa proses inisialisasi Sastrawi dimulai.
    logger.info("Menginisialisasi Sastrawi Stemmer dan StopWordRemover...")
    # Membuat pabrik (factory) untuk stemmer.
    factory = StemmerFactory()
    # Membuat objek stemmer dari factory.
    stemmer = factory.create_stemmer()

    # Membuat pabrik (factory) untuk stopword remover.
    stop_factory = StopWordRemoverFactory()
    # Mendapatkan daftar stopword (kata-kata umum seperti 'dan', 'di', 'yang').
    stopwords = stop_factory.get_stop_words()
    # Mengubah daftar stopword menjadi set untuk proses pencarian yang lebih cepat.
    stopwords_set = set(stopwords)
    # Memberi informasi bahwa inisialisasi Sastrawi telah selesai.
    logger.info("Inisialisasi Sastrawi selesai.")

    # Memberi informasi bahwa koneksi ke database sedang dibuat.
    logger.info("Membuat koneksi engine SQLAlchemy...")
    # Membangun string URL koneksi database dari konfigurasi yang diimpor.
    db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    # Membuat objek engine yang akan mengelola koneksi ke database.
    engine = create_engine(db_url)
    # Memberi informasi bahwa koneksi berhasil dibuat.
    logger.info("Koneksi engine berhasil dibuat.")

# Menangkap dan menangani jika terjadi kesalahan selama proses inisialisasi.
except Exception as e:
    # Mencatat pesan error yang terjadi.
    logger.error(f"Gagal saat inisialisasi awal: {e}")
    # Menghentikan program jika inisialisasi gagal.
    raise

# --- Kumpulan Fungsi Bantuan (Helper Functions) ---

# Fungsi untuk memuat kamus normalisasi dari file teks.
def load_normalization_dict(filepath):
    # Membuat dictionary kosong untuk menyimpan pasangan kata slang dan baku.
    norm_dict = {}
    # Mencatat informasi path file kamus yang sedang dibaca.
    logger.info(f"Membaca kamus normalisasi dari: {filepath}")
    try:
        # Membuka file kamus dengan encoding utf-8. 'with open' memastikan file ditutup secara otomatis.
        with open(filepath, 'r', encoding='utf-8') as f:
            # Membaca file baris per baris.
            for line in f:
                # Menghapus spasi atau karakter kosong di awal dan akhir baris.
                line = line.strip()
                # Memeriksa apakah baris tidak kosong dan mengandung pemisah tab.
                if line and '\t' in line:
                    # Memecah baris menjadi kata slang dan kata formal berdasarkan tab.
                    slang, formal = line.split('\t', 1)
                    # Menambahkan pasangan kata ke dalam dictionary.
                    norm_dict[slang] = formal
        # Memberi informasi jumlah kata yang berhasil dimuat.
        logger.info(f"Berhasil memuat {len(norm_dict)} kata dari kamus normalisasi.")
    # Menangani jika file kamus tidak ditemukan.
    except FileNotFoundError:
        logger.error(f"File kamus normalisasi tidak ditemukan di '{filepath}'. Normalisasi akan dilewati.")
    # Menangani kesalahan umum lainnya saat membaca file.
    except Exception as e:
        logger.error(f"Error saat membaca kamus normalisasi: {e}")
    # Mengembalikan dictionary yang sudah terisi.
    return norm_dict

# Fungsi untuk mengubah semua teks menjadi huruf kecil (case folding).
def case_folding(text):
    # Memeriksa apakah input adalah string, jika tidak, kembalikan string kosong.
    if not isinstance(text, str): return ""
    # Mengembalikan teks dalam format huruf kecil.
    return text.lower()

# Fungsi untuk membersihkan teks dari karakter yang tidak diinginkan (cleansing).
def cleansing(text):
    # Memeriksa apakah input adalah string.
    if not isinstance(text, str): return ""
    # Menghapus karakter non-ASCII untuk menghindari masalah encoding.
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Menghapus URL (http, https, www).
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Menghapus mention (@username).
    text = re.sub(r'@\w+', '', text)
    # Menghapus hashtag (#topic).
    text = re.sub(r'#\w+', '', text)
    # Menghapus semua karakter selain huruf, angka, dan spasi (termasuk tanda baca).
    text = re.sub(r'[^\w\s]', ' ', text)
    # Menghapus semua angka.
    text = re.sub(r'\d+', '', text)
    # Mengganti spasi ganda atau lebih menjadi satu spasi dan menghapus spasi di awal/akhir.
    text = re.sub(r'\s+', ' ', text).strip()
    # Mengembalikan teks yang sudah bersih.
    return text

# Fungsi untuk memecah kalimat menjadi daftar kata (tokenizing).
def tokenize(text):
    # Memeriksa apakah input adalah string yang tidak kosong.
    if not isinstance(text, str) or text.strip() == '': return []
    # Memecah string menjadi list kata berdasarkan spasi.
    return text.split()

# Fungsi untuk menormalisasi token menggunakan kamus yang telah dimuat.
def normalize_tokens(tokens, norm_dict):
    # Memeriksa apakah input adalah sebuah list.
    if not isinstance(tokens, list): return []
    # Mengganti setiap token dengan versi formalnya jika ada di kamus, jika tidak, gunakan token asli.
    return [norm_dict.get(token, token) for token in tokens]

# Fungsi untuk menghapus stopwords dari daftar token.
def remove_stopwords(tokens):
    # Memeriksa apakah input adalah sebuah list.
    if not isinstance(tokens, list): return []
    return [word for word in tokens if word not in stopwords_set]

# Fungsi untuk mengubah setiap token menjadi kata dasarnya (stemming).
def apply_stemmer(tokens):
    # Memeriksa apakah input adalah sebuah list.
    if not isinstance(tokens, list): return []
    return [stemmer.stem(word) for word in tokens]

# --- Fungsi Utama Preprocessing ---

# Fungsi utama yang menjalankan seluruh alur kerja preprocessing.
def run_preprocessing():
    try:
        # Mencatat waktu mulai proses.
        start_time = time.time()
        # Mencetak header untuk menandakan proses dimulai.
        logger.info("="*50)
        logger.info("MEMULAI PROSES PREPROCESSING DATA")
        logger.info("="*50)

        # Langkah 0: Membaca data mentah dari database.
        logger.info("Langkah 0: Membaca data dari database...")
        # Query SQL untuk mengambil kolom yang dibutuhkan dari tabel.
        query = "SELECT full_text, sentiment_pakar FROM data_efesiensi"
        # Menjalankan query dan memuat hasilnya ke dalam DataFrame pandas.
        df = pd.read_sql(query, engine)
        # Mencatat jumlah baris data yang berhasil dibaca.
        logger.info(f"Berhasil membaca {len(df)} baris data mentah dari tabel.")

        # Memeriksa apakah DataFrame kosong setelah dibaca.
        if df.empty:
            # Jika kosong, beri peringatan dan hentikan proses.
            logger.warning("Tidak ada data di tabel 'data_efesiensi' untuk diproses.")
            logger.warning("Proses preprocessing dihentikan.")
            # Mengembalikan status kegagalan.
            return {
                'success': False,
                'error': "Tidak ada data di tabel 'data_efesiensi' untuk diproses."
            }
        
        # Menampilkan 2 baris pertama dari data mentah sebagai sampel.
        logger.info("Contoh data mentah:\n" + df.head(2).to_string())

        # Membersihkan DataFrame dari baris yang kosong dan duplikat teks asli.
        df = df.dropna(subset=['full_text'])
        df = df[df['full_text'].astype(str).str.strip() != '']
        
        count_before_deduplication = len(df)
        logger.info(f"Jumlah data sebelum penghapusan duplikat teks asli: {count_before_deduplication}")
        df = df.drop_duplicates(subset=['full_text'])
        duplicates_removed = count_before_deduplication - len(df)
        logger.info(f"Menghapus {duplicates_removed} duplikat berdasarkan teks asli.")
        logger.info(f"Jumlah data setelah pembersihan awal: {len(df)}")

        # Memeriksa lagi jika DataFrame menjadi kosong setelah dibersihkan.
        if df.empty:
            logger.warning("Setelah dibersihkan, tidak ada data valid yang tersisa untuk diproses.")
            return {
                'success': False,
                'error': 'Setelah dibersihkan, tidak ada data valid yang tersisa.'
            }

        # Menentukan indeks sampel untuk ditampilkan di log.
        sample_index = 0 if not df.empty else None

        # --- Tahap 1: Case Folding ---
        logger.info("\n--- Langkah 1: Case Folding ---")
        df['case_folding'] = df['full_text'].apply(case_folding)
        if sample_index is not None:
            logger.info("Contoh: " + df['case_folding'].iloc[sample_index])

        # --- Tahap 2: Cleansing ---
        logger.info("\n--- Langkah 2: Cleansing ---")
        df['cleansing'] = df['case_folding'].apply(cleansing)
        if sample_index is not None:
            logger.info("Contoh: " + df['cleansing'].iloc[sample_index])

        # --- Tahap 3: Tokenizing ---
        logger.info("\n--- Langkah 3: Tokenizing ---")
        df['tokens'] = df['cleansing'].apply(tokenize)
        if sample_index is not None:
            logger.info("Contoh: " + str(df['tokens'].iloc[sample_index]))

        # --- Tahap 4: Normalisasi ---
        logger.info("\n--- Langkah 4: Normalisasi ---")
        norm_dict = load_normalization_dict('kamus/normalisasi.txt')
        df['normalized_tokens'] = df['tokens'].apply(lambda t: normalize_tokens(t, norm_dict))
        if sample_index is not None:
            logger.info("Contoh: " + str(df['normalized_tokens'].iloc[sample_index]))

        # --- Tahap 5: Stopword Removal ---
        logger.info("\n--- Langkah 5: Stopword Removal ---")
        df['stopwords_tokens'] = df['normalized_tokens'].apply(remove_stopwords)
        if sample_index is not None:
            logger.info("Contoh: " + str(df['stopwords_tokens'].iloc[sample_index]))

        # --- Tahap 6: Stemming ---
        logger.info("\n--- Langkah 6: Stemming ---")
        df['stemming_tokens'] = df['stopwords_tokens'].apply(apply_stemmer)
        if sample_index is not None:
            logger.info("Contoh: " + str(df['stemming_tokens'].iloc[sample_index]))

        # --- Tahap 7: Menggabungkan kembali token menjadi string ---
        logger.info("\n--- Langkah 7: Menggabungkan token menjadi string ---")
        df['tokenizing'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        df['normalized'] = df['normalized_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        df['stopwords'] = df['stopwords_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        df['stemming'] = df['stemming_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        logger.info("Penggabungan token selesai.")

        # --- PERUBAHAN BARU: Menampilkan detail duplikat SEBELUM menghapus ---
        logger.info("\n--- Menganalisis duplikat berdasarkan hasil stemming ---")
        
        # Cari semua baris yang merupakan duplikat berdasarkan kolom 'stemming'
        # keep=False menandai SEMUA kemunculan duplikat sebagai True
        is_duplicate = df.duplicated(subset=['stemming'], keep=False)
        
        # Filter DataFrame untuk hanya mendapatkan baris-baris yang duplikat
        duplicated_df = df[is_duplicate]
        
        if not duplicated_df.empty:
            # Hitung jumlah unik duplikat untuk dilaporkan
            num_duplicates_to_remove = len(duplicated_df[duplicated_df.duplicated(subset=['stemming'])])
            logger.warning(f"Ditemukan {num_duplicates_to_remove} baris duplikat yang akan dihapus. Rincian:")
            
            # Kelompokkan berdasarkan teks stemming untuk menampilkannya bersama-sama
            for stemmed_text, group in duplicated_df.groupby('stemming'):
                # Hanya tampilkan jika ada lebih dari satu teks asli dalam grup
                if len(group) > 1:
                    logger.info(f"\n- Teks Stemming Duplikat: '{stemmed_text}'")
                    logger.info("  Dihasilkan dari Teks Asli berikut:")
                    for i, original_text in enumerate(group['full_text'], 1):
                        logger.info(f"    {i}. {original_text}")
        else:
            logger.info("Tidak ada duplikat yang ditemukan berdasarkan hasil stemming.")
        # --- AKHIR PERUBAHAN BARU ---

        # Lanjutkan dengan proses penghapusan duplikat
        count_before_stem_deduplication = len(df)
        # df = df.drop_duplicates(subset=['stemming'], keep='first')
        count_after_stem_deduplication = len(df)
        stem_duplicates_removed = count_before_stem_deduplication - count_after_stem_deduplication
        logger.info(f"\nProses penghapusan: {stem_duplicates_removed} duplikat berdasarkan hasil stemming telah dihapus.")
        logger.info(f"Jumlah data final: {count_after_stem_deduplication}")
        
        # --- Tahap 8: Menyimpan hasil ke database ---
        logger.info("\n--- Langkah 8: Menyimpan hasil ke database ---")
        # Memilih kolom-kolom yang relevan untuk disimpan ke database.
        result_df = df[[
            'full_text', 'case_folding', 'cleansing', 'tokenizing',
            'normalized', 'stopwords', 'stemming', 'sentiment_pakar'
        ]]
                
        # Menyimpan DataFrame hasil ke tabel 'preprocessing' di database.
        result_df.to_sql(
            name='preprocessing',  # Nama tabel tujuan.
            con=engine,  # Engine koneksi database yang digunakan.
            if_exists='replace',  # Jika tabel sudah ada, ganti dengan yang baru.
            index=False  # Tidak menyertakan indeks DataFrame sebagai kolom di tabel.
        )
        # Mencatat informasi jumlah baris yang berhasil disimpan.
        logger.info(f"Berhasil menyimpan {len(result_df)} baris ke tabel 'preprocessing'.")

        # Mencatat waktu selesai proses.
        end_time = time.time()
        # Menghitung total waktu pemrosesan.
        processing_time = round(end_time - start_time, 2)
        # Mencetak footer sebagai penanda proses selesai beserta total waktunya.
        logger.info("="*50)
        logger.info(f"PROSES PREPROCESSING SELESAI dalam {processing_time} detik.")
        logger.info("="*50)

        # Mengembalikan dictionary yang berisi ringkasan hasil proses.
        return {
            'success': True,
            'original_count': count_before_deduplication, # Menggunakan hitungan sebelum duplikasi awal
            'final_count': len(result_df),
            'time_taken': processing_time
        }

    # Menangkap dan menangani kesalahan fatal yang mungkin terjadi selama proses utama.
    except Exception as e:
        # Mencatat error beserta traceback-nya untuk debugging.
        logger.error(f"Error fatal saat preprocessing: {e}", exc_info=True)
        # Mengembalikan status kegagalan beserta pesan error.
        return {'success': False, 'error': str(e)}

# Blok ini akan dieksekusi hanya jika skrip dijalankan secara langsung (bukan diimpor sebagai modul).
if __name__ == '__main__':
    # Memanggil fungsi utama untuk memulai proses preprocessing.
    run_preprocessing()
