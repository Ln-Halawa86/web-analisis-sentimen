from flask import Flask, render_template, request, redirect, url_for, flash,session,jsonify,Response
from flask_mysqldb import MySQL
import os
import pandas as pd
import mysql.connector as mysql2
import numpy as np
from subprocess import call
from config import DB_CONFIG
from sqlalchemy import create_engine, text
from subprocess import call
from preprocessing import run_preprocessing
from pembagian import split_data_logic
from collections import Counter
from ekstraksi import main_feature_extraction_pipeline
from model_naive_bayes import train_and_evaluate_nb
from werkzeug.utils import secure_filename
import io





app = Flask(__name__)
app.secret_key = 'many random bytes'

# Konfigurasi MySQL untuk Flask
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'skripsi4'

mysql = MySQL(app)

# mengaktifkan auto reload
app.config["TEMPLATES_AUTO_RELOAD"] = True
# mengaktifkan mode debug
app.config["DEBUG"] = True
# folder untuk upload file
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER2 = 'uploads'
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

from auth import auth_bp
app.register_blueprint(auth_bp)

from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Silakan login terlebih dahulu.")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def Index():
    return render_template('dashboard.html')

# dataset

def parse_data_file(filePath):
    """
    Fungsi untuk membaca file CSV atau Excel dan memasukkan datanya ke database.
    Versi ini sudah disesuaikan dengan struktur tabel Anda dan ditambahkan DEBUGGING.
    """
    print("\n--- MULAI PROSES PARSING ---")
    print(f"DEBUG: Memparsing file di: {filePath}")
    
    _, file_extension = os.path.splitext(filePath)
    
    try:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(filePath, header=0, encoding='unicode_escape')
        elif file_extension.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(filePath, header=0)
        else:
            flash("Format file tidak didukung. Harap unggah file CSV atau Excel.", "danger")
            return

        print("DEBUG: File berhasil dibaca oleh pandas.")
        print("DEBUG: 5 baris pertama dari file:")
        print(df.head())
        print("\nDEBUG: Nama kolom yang terdeteksi di file:")
        print(df.columns.tolist())

        # Mengganti nama kolom dari file agar sesuai dengan yang dibutuhkan kode
        if 'created_at' in df.columns:
            df.rename(columns={'created_at': 'tgl_tweet'}, inplace=True)
            print("DEBUG: Nama kolom 'conversation_created_a' berhasil diubah menjadi 'tgl_tweet'.")
            print("DEBUG: Nama kolom setelah diubah:")
            print(df.columns.tolist())

        # Validasi kolom yang dibutuhkan sesuai dengan database
        required_columns = ['tgl_tweet', 'full_text', 'username', 'sentiment_pakar']
        print(f"\nDEBUG: Memvalidasi kolom... Kolom yang dibutuhkan: {required_columns}")
        
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"--- ERROR: Validasi gagal. Kolom yang hilang: {missing_cols} ---")
            flash(f"File Anda kekurangan kolom yang wajib ada: {', '.join(missing_cols)}", "danger")
            return
        
        print("DEBUG: Validasi kolom berhasil.")

        # PERBAIKAN: Ganti nilai kosong (NaN) dengan None agar kompatibel dengan database
        # Metode astype(object) lebih andal untuk memastikan NaN diubah menjadi None
        df = df.astype(object).where(pd.notnull(df), None)
        print("DEBUG: Data file berhasil dimuat dan nilai NaN diganti dengan None.")
    except Exception as e:
        print(f"--- ERROR saat membaca atau memvalidasi file: {e} ---")
        flash(f"Gagal membaca file. Pastikan formatnya benar. Error: {e}", "danger")
        return

    values = []
    print("\nDEBUG: Memulai iterasi baris untuk dimasukkan ke database...")
    for i, row in df.iterrows():
        if isinstance(row.get('full_text'), str):
            values.append((
                row.get('tgl_tweet'),
                row.get('full_text'),
                row.get('username'),
                row.get('sentiment_pakar')
            ))
        if i < 3: # Cetak 3 baris pertama untuk sampel
             print(f"  - Baris {i} diproses: {values[-1] if values else 'Data tidak valid'}")


    if not values:
        print("--- ERROR: Tidak ada data valid yang ditemukan untuk dimasukkan ke database. ---")
        flash("Tidak ada data valid yang ditemukan di dalam file.", "warning")
        return

    print(f"\nDEBUG: Total {len(values)} baris data siap untuk dimasukkan ke database.")
    print("DEBUG: Contoh 3 baris pertama yang akan dimasukkan:")
    print(values[:3])

    try:
        print("\nDEBUG: Mencoba menghubungkan ke database untuk INSERT...")
        cur = mysql.connection.cursor()
        print("DEBUG: Menjalankan TRUNCATE TABLE data_efesiensi...")
        cur.execute("TRUNCATE TABLE data_efesiensi")
        
        sql = """
        INSERT INTO data_efesiensi (tgl_tweet, full_text, username, sentiment_pakar)
        VALUES (%s, %s, %s, %s)
        """
        
        print("DEBUG: Menjalankan cur.executemany(...)")
        cur.executemany(sql, values)
        mysql.connection.commit()
        cur.close()
        print("--- PROSES PARSING SELESAI SUKSES ---")
        flash(f"{len(values)} baris data berhasil diproses dan disimpan ke database.", "success")
    except Exception as e:
        mysql.connection.rollback()
        print(f"--- ERROR saat memasukkan data ke database: {e} ---")
        flash(f"Terjadi kesalahan saat menyimpan ke database: {e}", "danger")

def allowed_file(filename):
    """Fungsi untuk memeriksa ekstensi file yang diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx'}

@app.route('/dataset', methods=['GET', 'POST'])
@login_required # Mengaktifkan kembali decorator login
def dataset_handler():
    """
    Satu fungsi untuk menangani GET (menampilkan data) dan POST (mengunggah file).
    """
    # Logika untuk metode POST (mengunggah file)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada bagian file dalam request.', 'danger')
            return redirect(request.url)

        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            flash('Tidak ada file yang dipilih.', 'warning')
            return redirect(request.url)

        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            
            # Panggil fungsi parsing yang baru
            parse_data_file(file_path)
            
            return redirect(url_for('dataset_handler'))
        else:
            flash('Format file tidak valid. Harap unggah file .csv, .xls, atau .xlsx', 'danger')
            return redirect(request.url)

    # Logika untuk metode GET (menampilkan data)
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, tgl_tweet, full_text, username, sentiment_pakar FROM data_efesiensi")
    data = cur.fetchall()
    cur.close()
    return render_template('dataset.html', data_efesiensi=data)

@app.route('/delete_dataset1', methods=['POST'])
@login_required
def delete_dataset():
    """
    Menghapus semua data dari tabel data_efesiensi menggunakan TRUNCATE.
    """
    try:
        cur = mysql.connection.cursor()
        # TRUNCATE TABLE lebih cepat dan efisien untuk mengosongkan tabel
        cur.execute("TRUNCATE TABLE data_efesiensi")
        mysql.connection.commit()
        cur.close()
        # Beri pesan sukses kepada pengguna
        flash("Semua data dari dataset berhasil dihapus.", "success")
    except Exception as e:
        # Jika terjadi error, beri pesan gagal
        flash(f"Gagal menghapus dataset: {e}", "danger")
    
    # Arahkan kembali ke halaman dataset setelah selesai
    return redirect(url_for('dataset_handler'))
   
# end database


#logikan tampilan pelabelan pakar

@app.route('/labelpakar')
@login_required
def label():
    """
    Menampilkan halaman pelabelan dengan data yang sudah diproses untuk tampilan.
    """
    try:
        cur = mysql.connection.cursor()
        
        # Ambil semua data yang dibutuhkan dari database
        cur.execute("SELECT id, full_text, sentiment_pakar, status_validasi FROM data_efesiensi")
        raw_data = cur.fetchall()
        cur.close()

        # --- PERUBAHAN LOGIKA: Proses data di backend ---
        processed_data = []
        for row in raw_data:
            # PERUBAHAN: Status validasi selalu dianggap TRUE untuk keperluan tampilan
            is_validated = True
            
            processed_data.append({
                'id': row[0],
                'full_text': row[1],
                'sentiment_pakar': row[2],
                'is_validated': is_validated # Kirim status boolean ke template
            })
        
        # --- Hitung statistik dari data mentah ---
        stats = {
            'total': len(raw_data), 'positif': 0, 'negatif': 0, 'netral': 0, 'belum_dilabeli': 0
        }
        
        for row in raw_data:
            sentiment = row[2] # Indeks 2 adalah sentiment_pakar
            if sentiment == 'positif':
                stats['positif'] += 1
            elif sentiment == 'negatif':
                stats['negatif'] += 1
            elif sentiment == 'netral':
                stats['netral'] += 1
            else:
                stats['belum_dilabeli'] += 1
        
        # Kirim data yang sudah diproses dan statistik ke template
        return render_template('label.html', data_efesiensi=processed_data, stats=stats)
        
    except Exception as e:
        print(f"ERROR: Terjadi exception di /labelpakar: {e}")
        flash(f"Terjadi kesalahan saat mengambil data: {e}")
        return render_template('label.html', data_efesiensi=[], stats=None)


@app.route('/validate', methods=['POST'])
# @login_required
def validate_data():
    """
    API Endpoint untuk MENYIMPAN validasi dari pakar dengan debugging lengkap.
    """
    print("\n--- DEBUG: Memasuki /validate ---")
    try:
        # 1. Debug data yang diterima dari frontend
        print("DEBUG: Mencoba mengambil data JSON dari request...")
        updates = request.get_json()
        if not updates:
            print("ERROR: Tidak ada data JSON yang diterima.")
            return jsonify({'status': 'error', 'message': 'No JSON data received'}), 400
        print(f"DEBUG: Data JSON diterima: {updates}")

        cur = mysql.connection.cursor()
        print("DEBUG: Koneksi DB berhasil, cursor dibuat.")
        
        # Siapkan query UPDATE
        sql = """
        UPDATE data_efesiensi 
        SET sentiment_pakar = %s, status_validasi = %s 
        WHERE id = %s
        """
        
        # 2. Debug nilai yang akan di-update
        update_values = []
        for item in updates:
            # (sentiment_baru, status_baru, id_baris)
            update_values.append((item['sentiment'], True, item['id']))
        
        print(f"DEBUG: Nilai yang akan di-update ke DB: {update_values}")
        
        # Eksekusi semua pembaruan sekaligus
        print(f"DEBUG: Menjalankan executemany dengan {len(update_values)} nilai...")
        cur.executemany(sql, update_values)
        print("DEBUG: executemany berhasil.")
        
        mysql.connection.commit()
        print("DEBUG: Perubahan berhasil di-commit ke database.")
        cur.close()
        
        # 3. Debug respons yang dikirim kembali
        response = {'status': 'success', 'message': f'{len(updates)} data berhasil divalidasi.'}
        print(f"DEBUG: Mengirim respons sukses: {response}")
        return jsonify(response)

    except Exception as e:
        # 4. Debug jika terjadi error
        print(f"---!!! ERROR di /validate: {e} !!!---")
        import traceback
        traceback.print_exc() # Mencetak traceback lengkap untuk analisis mendalam
        
        # Coba rollback jika terjadi error
        try:
            mysql.connection.rollback()
            print("DEBUG: Rollback database berhasil.")
        except Exception as rb_e:
            print(f"ERROR: Gagal melakukan rollback: {rb_e}")

        return jsonify({'status': 'error', 'message': str(e)}), 500


# PASTIKAN FUNGSI INI ADA DI DALAM app.py ANDA
@app.route('/reset_validation', methods=['POST'])
# @login_required
def reset_validation():
    """
    API Endpoint untuk MENGHAPUS/RESET validasi pakar dengan debugging lengkap.
    """
    print("\n--- DEBUG: Memasuki /reset_validation ---")
    try:
        # 1. Debug data yang diterima
        print("DEBUG: Mencoba mengambil data JSON dari request...")
        data = request.get_json()
        if not data or 'id' not in data:
            print("ERROR: Data JSON tidak valid atau tidak ada 'id'.")
            return jsonify({'status': 'error', 'message': "Invalid JSON data, 'id' is required."}), 400
            
        item_id = data['id']
        print(f"DEBUG: Data JSON diterima. ID untuk direset: {item_id}")
        
        cur = mysql.connection.cursor()
        print("DEBUG: Koneksi DB berhasil, cursor dibuat.")
        
        # Query untuk mereset sentimen dan status validasi
        sql = """
        UPDATE data_efesiensi 
        SET sentiment_pakar = NULL, status_validasi = FALSE 
        WHERE id = %s
        """
        
        # 2. Debug query dan nilai yang akan dieksekusi
        print(f"DEBUG: Menjalankan query: {sql.strip()} dengan nilai: ({item_id},)")
        cur.execute(sql, (item_id,))
        print("DEBUG: Query eksekusi berhasil.")

        mysql.connection.commit()
        print("DEBUG: Perubahan berhasil di-commit ke database.")
        cur.close()
        
        # 3. Debug respons yang dikirim kembali
        response = {'status': 'success', 'message': f'Validasi untuk ID {item_id} berhasil direset.'}
        print(f"DEBUG: Mengirim respons sukses: {response}")
        return jsonify(response)
        
    except Exception as e:
        # 4. Debug jika terjadi error
        print(f"---!!! ERROR di /reset_validation: {e} !!!---")
        import traceback
        traceback.print_exc()

        try:
            mysql.connection.rollback()
            print("DEBUG: Rollback database berhasil.")
        except Exception as rb_e:
            print(f"ERROR: Gagal melakukan rollback: {rb_e}")
            
        return jsonify({'status': 'error', 'message': str(e)}), 500

#end pelabelan pakar

#star preprocessing

# Route untuk menampilkan halaman HTML
@app.route('/preprocessing')
@login_required
def preprocessing_page():
    """Hanya menampilkan halaman preprocessing.html."""
    return render_template('preprocessing.html')

# Endpoint untuk MENJALANKAN proses preprocessing
@app.route('/jalankan-preprocessing', methods=['POST'])
@login_required
def jalankan_preprocessing_route():
    """API Endpoint untuk MENJALANKAN skrip preprocessing."""
    try:
        hasil = run_preprocessing()
        if hasil.get('success'):
            return jsonify({'status': 'success', 'message': 'Proses preprocessing berhasil dijalankan.'})
        else:
            error_message = hasil.get('error', 'Kesalahan tidak diketahui di skrip preprocessing.')
            return jsonify({'status': 'error', 'message': error_message}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal memanggil skrip: {str(e)}'}), 500

# Endpoint untuk MENGAMBIL data hasil proses
@app.route('/get-processed-data', methods=['GET'])
@login_required
def get_processed_data_route():
    """API Endpoint untuk MENGAMBIL data dari tabel 'preprocessing'."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT full_text, case_folding, cleansing, tokenizing, normalized, stopwords, stemming FROM preprocessing")
        column_names = [desc[0] for desc in cur.description]
        data = [dict(zip(column_names, row)) for row in cur.fetchall()]
        cur.close()
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal mengambil data: {str(e)}'}), 500

# Endpoint untuk MENGHAPUS data hasil proses
@app.route('/hapus-data-preprocessing', methods=['POST'])
@login_required
def hapus_data_preprocessing_route():
    """API Endpoint untuk MENGHAPUS semua data dari tabel 'preprocessing'."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("TRUNCATE TABLE preprocessing")
        mysql.connection.commit()
        cur.close()
        return jsonify({'status': 'success', 'message': 'Data hasil preprocessing berhasil dihapus.'})
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'status': 'error', 'message': f'Gagal menghapus data: {str(e)}'}), 500

# Route untuk MENGUNDUH data sebagai CSV
@app.route('/download-preprocessing-csv')
@login_required
def download_preprocessing_csv_route():
    """API Endpoint untuk mengunduh semua data dari tabel 'preprocessing' sebagai file CSV."""
    try:
        # Menggunakan koneksi dari objek mysql yang sudah ada
        conn = mysql.connection
        
        # Membaca seluruh data dari tabel 'preprocessing' menggunakan pandas
        df = pd.read_sql("SELECT * FROM preprocessing", conn)
        
        # Membuat file CSV di dalam memori (tanpa menyimpan ke disk server)
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        csv_data = output.getvalue()
        
        # Membuat dan mengirim response sebagai file yang bisa di-download
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename=hasil_preprocessing.csv"}
        )
    except Exception as e:
        print(f"Error saat membuat CSV: {e}") # Mencatat error di log server
        return "Gagal membuat file CSV.", 500

# @app.route('/label')
# def label():
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT * FROM label")
#     data = cur.fetchall()
#     cur.close()
#     stats = session.get('label_stats', None)
#     session.pop('label_stats', None)  # Hapus setelah ditampilkan
#     return render_template('label.html', label=data, stats=stats)

# @app.route('/proses_label')
# def proses_label():
#     result = jalankan_pelabelan()
#     if result.get('success'):
#         flash("Pelabelan selesai!")
#         session['label_stats'] = result
#     else:
#         flash("Gagal menjalankan pelabelan.")
#     return redirect(url_for('label'))

# @app.route('/hapus_label')
# def hapus_label():
#     cur = mysql.connection.cursor()
#     cur.execute("TRUNCATE TABLE label")
#     mysql.connection.commit()
#     cur.close()
#     flash("Data label berhasil dihapus.")
#     return redirect(url_for('label'))


@app.route('/ekstraksi')
def ekstraksi_page():
    return render_template('ekstraksi.html')

@app.route('/proses-ekstraksi', methods=['POST'])
@login_required
def proses_ekstraksi_route():
    """
    API Endpoint untuk menjalankan proses ekstraksi.
    Menerima parameter 'use_smote' untuk menentukan apakah akan menggunakan SMOTE.
    """
    try:
        # Mengambil parameter dari body request JSON
        use_smote = request.json.get('use_smote', False)
        
        # Memanggil pipeline backend dengan parameter yang sesuai
        full_result = main_feature_extraction_pipeline(use_smote=use_smote)

        if not full_result.get('success'):
            return jsonify(full_result), 500

        # Membuat respons yang hanya berisi data yang dibutuhkan oleh UI
        web_response = {
            'success': True,
            'stats': full_result.get('stats', {}),
        }
        return jsonify(web_response)
    except Exception as e:
        # Mencatat error di log server untuk debugging
        app.logger.error(f"Error di route /proses-ekstraksi: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# Ganti route '/hapus_ekstraksi' yang lama dengan versi API ini
@app.route('/hapus-ekstraksi', methods=['POST'])
@login_required
def hapus_ekstraksi_route():
    """
    API Endpoint untuk menghapus hasil ekstraksi (file model & vectorizer).
    """
    try:
        # Hapus file vectorizer jika ada
        if os.path.exists('tfidf_vectorizer.pkl'):
            os.remove('tfidf_vectorizer.pkl')
            
        # Hapus juga file model jika ada (asumsi nama file)
        if os.path.exists('model_sentimen.pkl'):
            os.remove('model_sentimen.pkl')
        
        # Mengosongkan session jika Anda menyimpan sesuatu di sana
        session.pop('extractionResult', None)
        
        return jsonify({'status': 'success', 'message': 'Hasil ekstraksi berhasil dihapus.'})
    except Exception as e:
        app.logger.error(f"Error di route /hapus-ekstraksi: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

# app.py

@app.route('/pembagian_data')
def show_splitting_page():
    # Fungsi ini hanya menampilkan halaman HTML-nya
    return render_template('pembagian_data.html') # Ganti dengan nama file HTML Anda

# INI ADALAH API ENDPOINT YANG SUDAH DIPERBARUI DAN LEBIH BERSIH
@app.route('/proses-pembagian', methods=['POST'])
def process_data_split():
    """
    API Endpoint yang dipanggil oleh JavaScript.
    Tugasnya sekarang hanya mengelola request dan response web.
    """
    try:
        # Menerima rasio dari frontend
        data = request.get_json()
        train_ratio_percent = int(data.get('ratio', 80))
        train_ratio_float = train_ratio_percent / 100.0

        # 2. Panggil fungsi logika bisnis untuk melakukan pekerjaan berat
        #    Fungsi ini akan mengambil data, membaginya, dan menyimpannya.
        #    (Asumsi 'mysql' adalah objek koneksi database Anda)
        result = split_data_logic(mysql.connection, train_ratio_float)

        # 3. Kirim hasil kembali ke frontend
        if result['status'] == 'success':
            return jsonify(result)
        else:
            # Kirim pesan error jika terjadi masalah di logika bisnis
            return jsonify({'status': 'error', 'message': result.get('message', 'Unknown error')}), 400

    except Exception as e:
        # Menangani error yang mungkin terjadi di level route (misal: JSON tidak valid)
        print(f"An error occurred in the route: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/hapus-pembagian-data', methods=['POST'])
def delete_split_data():
    """
    API Endpoint untuk menghapus semua data dari tabel pembagian_data.
    """
    try:
        cur = mysql.connection.cursor()
        # TRUNCATE TABLE lebih efisien daripada DELETE FROM untuk mengosongkan tabel
        cur.execute("TRUNCATE TABLE pembagian_data")
        mysql.connection.commit()
        cur.close()
        return jsonify({'status': 'success', 'message': 'Semua data dari tabel pembagian_data berhasil dihapus.'})
    except Exception as e:
        mysql.connection.rollback()
        print(f"Error saat menghapus data dari database: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500



# @app.route('/klasifikasi_naive_bayes')
# def klasifikasi_naive_bayes():
#     from klasifikasi import proses_klasifikasi
#     result = proses_klasifikasi()
#     if not result.get('success'):
#         flash("Gagal menjalankan klasifikasi.")
#         return redirect(url_for('dashboard'))
#     session['klasifikasi_stats'] = result
#     return redirect(url_for('hasil_klasifikasi'))

# @app.route('/hasil_klasifikasi')
# def hasil_klasifikasi():
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT * FROM hasil_klasifikasi ORDER BY id DESC LIMIT 1")
#     last_result = cur.fetchone()
#     cur.close()
#     return render_template('klasifikasi_naive_bayes.html', result=last_result)

# @app.route('/hapus_hasil_klasifikasi')
# def hapus_hasil_klasifikasi():
#     cur = mysql.connection.cursor()
#     cur.execute("TRUNCATE TABLE hasil_klasifikasi")
#     mysql.connection.commit()
#     cur.close()
#     flash("Data hasil klasifikasi berhasil dihapus.")
#     return redirect(url_for('hasil_klasifikasi'))
    


# --- HALAMAN UTAMA (RENDER HTML) ---
# Ganti nama fungsi ini jika di kode Anda berbeda
@app.route('/klasifikasi')
@login_required
def klasifikasi_page():
    """Menampilkan halaman untuk pelatihan dan evaluasi model."""
    return render_template('klasifikasi_naive_bayes1.html')

# API Endpoint untuk menjalankan seluruh alur (ekstraksi + pelatihan)
@app.route('/latih-dan-evaluasi', methods=['POST'])
@login_required
def latih_dan_evaluasi_route():
    """
    API Endpoint yang dipanggil oleh frontend untuk melatih dan mengevaluasi model.
    Menerima 'use_smote' sebagai parameter.
    """
    try:
        # 1. Ambil parameter dari request
        use_smote = request.json.get('use_smote', False)
        
        # 2. Jalankan pipeline ekstraksi fitur dari ekstraksi.py
        # Parameter use_smote akan menentukan apakah SMOTE dijalankan atau tidak
        feature_results = main_feature_extraction_pipeline(use_smote=use_smote)
        if not feature_results.get('success'):
            # Jika ekstraksi gagal, langsung kembalikan errornya
            return jsonify(feature_results), 500

        # 3. Jalankan pipeline pelatihan dan evaluasi dari model_naive_bayes.py
        # Berikan hasil dari langkah 2 sebagai input
        evaluation_results = train_and_evaluate_nb(feature_results)
        
        # Hapus objek model yang tidak bisa dikirim sebagai JSON
        if 'model' in evaluation_results:
            del evaluation_results['model']
            
        return jsonify(evaluation_results)

    except Exception as e:
        # Menangani error tak terduga
        app.logger.error(f"Error di route /latih-dan-evaluasi: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/hapus_hasil_pelatihan', methods=['POST'])
def hapus_hasil_pelatihan():
    """
    API Endpoint untuk menghapus semua data dari tabel hasil_klasifikasi.
    """
    try:
        db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_url)
        with engine.connect() as connection:
            # Gunakan TRUNCATE TABLE untuk mereset tabel dengan cepat
            connection.execute(text("TRUNCATE TABLE hasil_klasifikasi"))
            # Perlu import text dari sqlalchemy: from sqlalchemy import text
        
        return jsonify({'success': True, 'message': 'Tabel hasil_klasifikasi berhasil dikosongkan.'})
    except Exception as e:
        # Log error di sisi server
        print(f"ERROR saat menghapus hasil pelatihan: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)