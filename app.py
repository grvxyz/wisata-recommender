from flask import Flask, render_template, request, abort, jsonify
import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from math import radians, sin, cos, sqrt, atan2
from flask import redirect, url_for

app = Flask(__name__)

# =====================================
# LOAD DATASET
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset-wisata-jogja-sekitar.csv")

df_all = pd.read_csv(DATASET_PATH)

# =====================================
# DATA CLEANING GLOBAL
# =====================================
df_all = df_all.drop_duplicates(subset=['nama']).reset_index(drop=True)

df_all['latitude'] = pd.to_numeric(df_all['latitude'], errors='coerce')
df_all['longitude'] = pd.to_numeric(df_all['longitude'], errors='coerce')

df_all['type'] = df_all['type'].fillna("").astype(str)
df_all['description'] = df_all['description'].fillna("").astype(str)
df_all['image'] = df_all.get('image', "").fillna("")
df_all['vote_average'] = df_all.get('vote_average', 0).fillna(0)
df_all['vote_count'] = df_all.get('vote_count', 0).fillna(0)

df_all['htm_weekday'] = pd.to_numeric(
    df_all.get('htm_weekday', 10000),
    errors='coerce'
).fillna(10000)

df_all['htm_weekend'] = pd.to_numeric(
    df_all.get('htm_weekend', 15000),
    errors='coerce'
).fillna(15000)

# =====================================
# DATA SIMULASI DASHBOARD
# =====================================
df_all['pengunjung'] = (df_all['vote_count'] * 10).astype(int)
df_all['pendapatan'] = df_all['pengunjung'] * df_all['htm_weekday']

# =====================================
# KLASIFIKASI AREA (BERDASAR GEO)
# =====================================
def klasifikasi_area(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "Tidak Diketahui"

    if -7.90 <= lat <= -7.70 and 110.30 <= lon <= 110.45:
        return "Kota Yogyakarta"
    elif -7.95 <= lat <= -7.55 and 110.25 <= lon <= 110.60:
        return "Sleman"
    elif -8.15 <= lat <= -7.75 and 110.15 <= lon <= 110.50:
        return "Bantul"
    elif -8.35 <= lat <= -7.80 and 110.40 <= lon <= 110.90:
        return "Gunungkidul"
    elif -8.30 <= lat <= -7.70 and 109.95 <= lon <= 110.35:
        return "Kulon Progo"
    else:
        return "Sekitar DIY"

df_all['area'] = df_all.apply(
    lambda x: klasifikasi_area(x['latitude'], x['longitude']),
    axis=1
)

# DATA KHUSUS REKOMENDASI (PRESISI TINGGI)

df_rec = df_all.dropna(subset=['latitude', 'longitude']).copy()
df_rec['id_all'] = df_rec.index   # SIMPAN ID ASLI

df_rec = df_rec[
    (df_rec['latitude'] >= -8.25) &
    (df_rec['latitude'] <= -7.50) &
    (df_rec['longitude'] >= 110.10) &
    (df_rec['longitude'] <= 110.85)
].reset_index(drop=True)

# TEXT PREPROCESSING

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df_rec['clean_description'] = df_rec['description'].apply(clean_text)
df_rec['content'] = df_rec['type'] + " " + df_rec['clean_description']

# =====================================
# TF-IDF MODEL
# =====================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_rec['content'])
distance_matrix = euclidean_distances(tfidf_matrix)

# =====================================
# FUNGSI HAVERSINE (JARAK GEO)
# =====================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
def evaluasi_precision_k(top_k=5):
    precision_list = []

    for idx in range(len(df_rec)):
        nama_query = df_rec.loc[idx, 'nama']
        type_query = df_rec.loc[idx, 'type']

        rekomendasi = rekomendasi_wisata(nama_query, top_k)
        if not rekomendasi:
            continue

        relevan = sum(
            1 for r in rekomendasi
            if r['type'] == type_query
        )

        precision = relevan / top_k
        precision_list.append(precision)

    return round(np.mean(precision_list) * 100, 2)

# =====================================
# REKOMENDASI (KATEGORI + GEO TERKUNCI)
# =====================================
def rekomendasi_wisata(nama, top_n=5):
    if nama not in df_rec['nama'].values:
        return []

    idx = df_rec.index[df_rec['nama'] == nama][0]

    lat_asal = df_rec.loc[idx, 'latitude']
    lon_asal = df_rec.loc[idx, 'longitude']
    type_asal = df_rec.loc[idx, 'type']

    kandidat = df_rec.copy()
    kandidat = kandidat[kandidat.index != idx]

    # =========================
    # 1. TF-IDF + EUCLIDEAN (45%)
    # =========================
    kandidat['jarak_konten'] = distance_matrix[idx][kandidat.index]
    kandidat['skor_konten'] = 1 / (1 + kandidat['jarak_konten'])

    # =========================
    # 2. FILTER KATEGORI (HARD CONSTRAINT)
    # =========================
    kandidat = kandidat[kandidat['type'] == type_asal]

    # =========================
    # 3. JARAK GEO (HAVERSINE) – 20%
    # =========================
    kandidat['jarak_geo'] = kandidat.apply(
        lambda x: haversine(
            lat_asal, lon_asal,
            x['latitude'], x['longitude']
        ),
        axis=1
    )
    kandidat['skor_geo'] = 1 / (1 + kandidat['jarak_geo'])

    # =========================
    # 4. RATING – 20%
    # =========================
    kandidat['skor_rating'] = kandidat['vote_average'] / 5

    # =========================
    # 5. POPULARITAS – 15%
    # =========================
    max_vote = df_rec['vote_count'].max()
    kandidat['skor_popularitas'] = (
        np.log1p(kandidat['vote_count']) / np.log1p(max_vote)
    )

    # =========================
    # SKOR FINAL (WEIGHTED)
    # =========================
    kandidat['score'] = (
        0.45 * kandidat['skor_konten'] +
        0.20 * kandidat['skor_geo'] +
        0.20 * kandidat['skor_rating'] +
        0.15 * kandidat['skor_popularitas']
    ).round(4)

    kandidat = kandidat.sort_values('score', ascending=False)

    hasil = kandidat.head(top_n).copy()
    hasil['id'] = hasil['id_all']


    return hasil.to_dict(orient='records')

# =====================================
# HOME (PILIH 1 WISATA SAJA)
# =====================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tempat = request.form.get("tempat")

        # cari ID wisata
        data = df_all[df_all['nama'] == tempat]
        if not data.empty:
            id_wisata = data.index[0]
            return redirect(url_for('detail', id=id_wisata))

    return render_template(
        "index.html",
        tempat_list=df_all['nama'].tolist()
    )


# =====================================
# DASHBOARD
# =====================================
@app.route("/dashboard")
def dashboard():
    data = df_all.copy()
    data['id'] = data.index

    return render_template(
        "dashboard.html",
        wisata=data.to_dict(orient='records'),
        total_destinasi=len(data),
        total_pengunjung=int(data['pengunjung'].sum()),
        total_pendapatan=int(data['pendapatan'].sum()),
        rating_rata=round(data['vote_average'].mean(), 2)
    )

# =====================================
# API FILTER DASHBOARD
# =====================================
@app.route("/api/wisata")
def api_wisata():
    kategori = request.args.get("kategori")
    rating = request.args.get("rating")
    harga = request.args.get("harga")
    area = request.args.get("area")
    sort = request.args.get("sort")

    data = df_all.copy()

    if kategori and kategori != "all":
        data = data[data['type'] == kategori]
    if area and area != "all":
        data = data[data['area'] == area]
    if rating:
        data = data[data['vote_average'] >= float(rating)]
    if harga:
        data = data[data['htm_weekday'] <= int(harga)]

    if sort == "rating_desc":
        data = data.sort_values("vote_average", ascending=False)
    elif sort == "rating_asc":
        data = data.sort_values("vote_average", ascending=True)

    data['id'] = data.index
    return jsonify(data.to_dict(orient="records"))

# =====================================
# DETAIL
# =====================================
@app.route("/wisata/<int:id>")
def detail(id):
    if id not in df_all.index:
        abort(404)

    wisata = df_all.loc[id]
    rekomendasi = rekomendasi_wisata(wisata['nama'], 4)

    skor_rating = (wisata['vote_average'] / 5) * 100

    skor_popularitas = min(
        np.log1p(wisata['vote_count']) /
        np.log1p(df_all['vote_count'].max()) * 100,
        100
    )

    if wisata['htm_weekday'] <= 20000:
        skor_harga = 100
    elif wisata['htm_weekday'] <= 50000:
        skor_harga = 70
    else:
        skor_harga = 40

    jarak_list = []
    for r in rekomendasi:
        d = haversine(
            wisata['latitude'], wisata['longitude'],
            r['latitude'], r['longitude']
        )
        jarak_list.append(d)

    skor_jarak = max(0, 100 - (np.mean(jarak_list) * 5)) if jarak_list else 50

    skor_final = round(
        (0.3 * skor_rating) +
        (0.25 * skor_popularitas) +
        (0.2 * skor_jarak) +
        (0.25 * skor_harga),
        2
    )

    return render_template(
        "detail.html",
        w=wisata,
        rekomendasi=rekomendasi,
        skor=skor_final
    )

# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    print("DATA DASHBOARD :", len(df_all))
    print("DATA REKOMENDASI:", len(df_rec))
    app.run(debug=True)
