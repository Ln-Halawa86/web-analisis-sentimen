import pandas as pd
from sklearn.model_selection import train_test_split

def _save_split_data_to_db(db_connection, train_data, test_data):
    """
    Fungsi internal untuk menyimpan hasil pembagian data ke database.
    Disesuaikan untuk menyertakan 'sentiment_pakar'.
    """
    try:
        # Tambahkan kolom 'data_type'
        train_data['data_type'] = 'training'
        test_data['data_type'] = 'testing'

        # Gabungkan kedua set data
        all_split_data = pd.concat([train_data, test_data], ignore_index=True)

        # --- PERUBAHAN 1: Ubah nama kolom agar sesuai dengan tabel tujuan ---
        all_split_data.rename(columns={
            'stemming': 'teks', 
            'sentiment': 'sentiment_pakar'
        }, inplace=True)
        
        # --- PERUBAHAN 2: Pilih kolom yang akan disimpan (termasuk sentiment_pakar) ---
        values = [tuple(x) for x in all_split_data[['teks', 'sentiment_pakar', 'data_type']].to_numpy()]

        cur = db_connection.cursor()
        
        # Kosongkan tabel sebelum memasukkan data baru
        print("DEBUG: Menjalankan TRUNCATE TABLE pembagian_data...")
        cur.execute("TRUNCATE TABLE pembagian_data")
        print("DEBUG: TRUNCATE TABLE berhasil.")
        
        # --- PERUBAHAN 3: Siapkan query SQL untuk menyertakan sentiment_pakar ---
        sql = """
        INSERT INTO pembagian_data (teks, sentiment_pakar, data_type) 
        VALUES (%s, %s, %s)
        """
        
        # Eksekusi query dengan semua data
        print(f"DEBUG: Menjalankan executemany dengan {len(values)} baris...")
        cur.executemany(sql, values)
        print("DEBUG: executemany berhasil.")

        db_connection.commit()
        cur.close()
        
        print("DEBUG: Penyimpanan ke database berhasil.")
        return {'status': 'success'}

    except Exception as e:
        db_connection.rollback()
        print(f"---!!! DEBUG: TERJADI ERROR saat menyimpan ke pembagian_data: {e} !!!---")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f"Gagal menyimpan ke database: {e}"}


def split_data_logic(db_connection, train_ratio_float):
    """
    Fungsi ini berisi logika inti untuk mengambil, membagi, dan menyimpan data.
    Disesuaikan dengan struktur tabel 'preprocessing' Anda.
    """
    try:
        # 1. Mengambil data dari tabel preprocessing
        cur = db_connection.cursor()
        cur.execute("SELECT stemming, sentiment_pakar FROM preprocessing WHERE sentiment_pakar IS NOT NULL AND sentiment_pakar != ''")
        all_data = cur.fetchall()
        cur.close()

        if not all_data:
            return {'status': 'error', 'message': 'Tidak ada data dengan label pakar yang ditemukan di tabel preprocessing.'}

        # Membuat DataFrame dengan nama kolom yang sesuai
        df = pd.DataFrame(all_data, columns=['stemming', 'sentiment'])

        # 2. Melakukan pembagian data
        X = df['stemming']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_ratio_float, 
            random_state=42,
            stratify=y
        )

        # Gabungkan kembali fitur (X) dan target (y)
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # 3. MENYIMPAN HASIL KE DATABASE
        save_result = _save_split_data_to_db(db_connection, train_data.copy(), test_data.copy())
        if save_result['status'] == 'error':
            return save_result

        # 4. Siapkan hasil untuk dikirim kembali ke frontend
        train_data_dict = train_data.to_dict('records')
        for i, row in enumerate(train_data_dict):
            row['id'] = i + 1
            row['text'] = row.pop('stemming')

        test_data_dict = test_data.to_dict('records')
        for i, row in enumerate(test_data_dict):
            row['id'] = i + 1
            row['text'] = row.pop('stemming')

        result = {
            'status': 'success',
            'training_data': train_data_dict,
            'testing_data': test_data_dict,
            'train_count': len(train_data),
            'test_count': len(test_data),
            'total_count': len(df)
        }
        return result

    except Exception as e:
        print(f"An error occurred in split_data_logic: {e}")
        return {'status': 'error', 'message': str(e)}
