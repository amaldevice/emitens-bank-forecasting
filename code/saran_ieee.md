# Strategi Pengembangan Proyek Untuk Publikasi IEEE

**Judul Potensial:** _A Hybrid Fast-Genetic Optimized LSTM with Macroeconomic Regressors for Predicting Indonesian Bank Stock Returns: An XAI-Based Investigation_

Dokumen ini berisi panduan komprehensif untuk meningkatkan kualitas teknis dan metodologis dari proyek LSTM Bank Indonesia agar memenuhi standar publikasi IEEE (Conference/Journal).

---

## 1. Peningkatan Arsitektur Model (Novelty)

Standard LSTM sudah dianggap "common practice". Untuk IEEE, diperlukan modifikasi atau kombinasi arsitektur:

- **Hybrid Architecture**: Implementasikan **CNN-LSTM** (CNN untuk feature extraction spasial, LSTM untuk temporal) atau **ConvLSTM**.
- **Attention Mechanism**: Tambahkan layer _Self-Attention_ atau _Bahdanau Attention_ setelah layer LSTM. Ini memungkinkan model "fokus" pada tanggal-tanggal tertentu yang memiliki volatilitas tinggi.
- **Bidirectional LSTM (Bi-LSTM)**: Gunakan Bi-LSTM untuk menangkap informasi dari masa depan (backpropagation dalam konteks time-series) yang seringkali memberikan hasil lebih stabil.

## 2. Feature Engineering & External Data

IEEE sangat menyukai model yang mempertimbangkan faktor eksternal riil (tidak hanya harga saham itu sendiri):

- **Sentimen Analisis**: Integrasikan skor sentimen dari berita keuangan (CNBC Indonesia, Kontan) atau media sosial (Twitter/Stockbit).
- **Faktor Makroekonomi**: Tambahkan variabel yang secara historis mempengaruhi perbankan Indonesia:
  - **BI 7-Day Reverse Repo Rate (BI7DRR)**: Suku bunga acuan.
  - **Inflasi (IHK)**.
  - **Kurs USD/IDR**.
  - **Indeks Sektoral Perbankan (IDXFINANCE)**.
- **Technical Indicators Berbasis Volume**: Tambahkan On-Balance Volume (OBV) atau Chaikin Money Flow untuk memperkuat sinyal akumulasi/distribusi.

## 3. Penguatan Metodologi Optimasi (FastGA)

Jika menggunakan "FastGA", pastikan ada justifikasi ilmiah yang kuat:

- **Benchmark Optimasi**: Bandingkan performa FastGA vs **Optuna (Bayesian Optimization)**, **PSO (Particle Swarm Optimization)**, atau **Grid Search**.
- **Analisis Konvergensi**: Sertakan grafik yang menunjukkan seberapa cepat FastGA menemukan parameter optimal dibanding metode lain.
- **Hiperparameter**: Jangan hanya mengoptimasi `units` dan `lr`, tambahkan `lookback_window` (60, 90, 120 hari) ke dalam pencarian GA.

## 4. Rigoritas Statistik & Validasi

Bagian ini sering menjadi alasan utama penolakan (reject) oleh reviewer:

- **Benchmark vs State-of-the-Art (SOTA)**: Bandingkan modelmu dengan model modern seperti:
  - **Informer** atau **Temporal Fusion Transformer (TFT)**.
  - **XGBoost/LightGBM** (sebagai baseline Machine Learning).
- **Ablation Study**: Lakukan pengujian komponen:
  - Test 1: LSTM saja.
  - Test 2: LSTM + GA.
  - Test 3: LSTM + GA + Macro Features.
  - Sajikan dalam tabel untuk membuktikan setiap fitur/metode memberikan kontribusi.
- **Statistical Significance Test**: Gunakan **Diebold-Mariano Test** untuk membuktikan bahwa keunggulan modelmu bukan karena faktor keberuntungan (stochastic nature).

## 5. Pendalaman Explainable AI (XAI)

Kamu sudah memiliki script XAI, sekarang pertegas narasinya:

- **Economic Correlation**: Hubungkan hasil SHAP/Permutation Importance dengan kejadian ekonomi nasional (misal: "Fitur High Price dominan saat pemulihan pasca-COVID, sedangkan Dividen dominan saat fase konsolidasi").
- **Local vs Global Explanation**: Gunakan SHAP untuk menjelaskan kenapa di hari tertentu (misal saat krisis mini) model memprediksi harga akan turun drastis.

## 6. Visualisasi & Format (Standar IEEE)

- **DPI Tinggi**: Pastikan semua grafik disimpan dengan minimal `dpi=300`.
- **Simbol Matematis**: Gunakan notasi formal untuk menjelaskan cara kerja LSTM dan proses Genetic Algorithm (bukan sekadar narasi kode).
- **Tabel Perbandingan**: Buat tabel ringkasan MSE, RMSE, MAE, dan MAPE untuk semua bank dalam satu tampilan yang rapi.

---

## Technical To-Do List (Action Plan)

### Sprint 1: Data Enrichment

- [ ] Buat script `04_extract_macro_macro.py` untuk ambil data historis BI Rate dan Inflasi.
- [ ] Gabungkan (merge) data makro tersebut ke dalam CSV bank masing-masing di folder `data/`.

### Sprint 2: Model Upgrading

- [ ] Modifikasi `train_price.py` untuk mendukung arsitektur Bi-LSTM dengan Attention.
- [ ] Implementasikan fungsi benchmark otomatis untuk membandingkan FastGA vs RandomSearch.

### Sprint 3: Statistical Rigor

- [ ] Tambahkan script khusus untuk **Diebold-Mariano Test** guna membandingkan signifikansi error antar model.
- [ ] Jalankan **Ablation Study** dan simpan semua hasilnya ke dalam folder `results/ablation/`.

### Sprint 4: Writing & Visualization

- [ ] Gunakan `seaborn` dengan tema `paper` untuk semua grafik hasil.
- [ ] Ekspor tabel performa ke format LaTeX (biasanya diminta untuk format IEEE).

---

**Catatan Jujur:** Proyekmu saat ini sudah mencapai 60-70% kelayakan. Tambahan di sisi **Macro Data** dan **Ablation Study** akan menjadi pendorong utama agar paper ini dianggap memiliki kontribusi ilmiah yang solid di mata reviewer IEEE.

---

## Reminder Implementasi (Disepakati)

Tambahan ini sebagai pengingat praktis agar eksperimen tetap ilmiah tetapi tidak "meledak" kombinasi:

- **Baseline SOTA minimum yang wajib ditambah**: `XGBoost` dan `LightGBM`.
- **Jumlah kombinasi baseline cukup 4 case inti**:
  - `xgb_data`
  - `xgb_macro`
  - `lgbm_data`
  - `lgbm_macro`
- **Tuning adil antar model**: gunakan budget yang sama untuk tiap model (contoh: 50 trial per model).
- **Replikasi konsisten**: pakai seed yang sama dengan studi LSTM/BiLSTM (contoh: 5 seed) agar uji statistik fair.
- **Evaluasi statistik**: lanjutkan Wilcoxon + Diebold-Mariano pada baseline terbaik vs model usulan.
- **Konsistensi narasi paper**: jika implementasi optimasi masih random search, jangan klaim sebagai FastGA sebelum FastGA benar-benar diimplementasikan.

Target praktis: tambah 4 baseline di atas + multi-seed + uji statistik sudah cukup kuat sebagai pembanding tanpa menambah kompleksitas berlebihan.
