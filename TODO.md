# TODO - Akurasi Model USD/IDR (Improved XGBoost)

## Step 1: Implement konsisten imputasi NaN (Selesai: Diganti ke Forward Fill)
- [x] Update `src/train_xgboost_improved.py` agar saat training melakukan imputasi NaN dengan `ffill()` secara berurutan pada data historis.
- [x] Hapus logika penyimpanan `median` ke metadata, karena `ffill` jauh lebih valid untuk data Time Series.

## Step 2: Validasi feature saat prediksi (Selesai)
- [x] Update `src/predict_latest_improved.py` agar sebelum predict:
  - cek `feature_cols` yang hilang dari data,
  - lakukan `ffill()` berurutan di sekuens data (agar hari terakhir mendapatkan harga hari sebelumnya jika NaN),
  - gunakan fallback 0 jika masih ada NaN yang tersisa.

## Step 3: Sinkronkan ke Streamlit (Selesai)
- [x] Update `app/streamlit_app.py` (fungsi improved model) agar imputasi dan validasi feature menggunakan metode `ffill()` yang sama dengan CLI.

## Step 4: Smoke test
- [ ] Jalankan training improved model untuk membuat ulang model + metadata.
- [ ] Jalankan `python src/predict_latest_improved.py`.
- [ ] Opsional: jalankan streamlit dan cek prediksi tampil.

