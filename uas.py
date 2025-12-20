import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Clustering Kemiskinan Jawa Barat",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown(
    """
    <h1 style='text-align:center;'>📊 Clustering Kemiskinan Jawa Barat</h1>
    <p style='text-align:center; font-size:16px;'>
    Analisis pengelompokan kabupaten/kota di Jawa Barat menggunakan
    <b>K-Means Clustering</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_jabar_kemiskinan.csv")

df = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Pengaturan")

k = st.sidebar.slider("Jumlah Cluster (k)", 2, 5, 3)

st.sidebar.markdown("### Filter Data")

tahun_list = sorted(df['tahun'].unique())
tahun_pilih = st.sidebar.multiselect("Tahun", tahun_list, default=tahun_list)

kab_list = sorted(df['nama_kabupaten_kota'].unique())
kab_pilih = st.sidebar.multiselect(
    "Kabupaten / Kota", kab_list, default=kab_list
)

df_f = df[
    (df['tahun'].isin(tahun_pilih)) &
    (df['nama_kabupaten_kota'].isin(kab_pilih))
]

if df_f.empty:
    st.warning("⚠️ Data kosong, silakan ubah filter.")
    st.stop()

# ======================================================
# METRIC CARD
# ======================================================
c1, c2, c3 = st.columns(3)
c1.metric("📌 Jumlah Data", len(df_f))
c2.metric("📍 Jumlah Wilayah", df_f['nama_kabupaten_kota'].nunique())
c3.metric("🧩 Jumlah Cluster", k)

st.divider()

# ======================================================
# VISUAL DISTRIBUSI DATA (EDA)
# ======================================================
st.subheader("📈 Distribusi Data Indikator")
st.caption("Visualisasi sebaran data sebelum proses clustering")

features = ['tpt', 'ppm', 'ppk', 'ipm', 'ikk']

fig_dist, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df_f[col], bins=10)
    axes[i].set_title(f'Distribusi {col}')
    axes[i].grid(alpha=0.3)

# kosongkan subplot terakhir
axes[-1].axis('off')

plt.tight_layout()
st.pyplot(fig_dist)

st.divider()

# ======================================================
# MODEL
# ======================================================
X = df_f[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42)
df_f['cluster'] = kmeans.fit_predict(X_scaled)

if k == 3:
    df_f['kategori_kemiskinan'] = df_f['cluster'].map({
        0: 'Kemiskinan Tinggi',
        1: 'Kemiskinan Sedang',
        2: 'Kemiskinan Rendah'
    })
else:
    df_f['kategori_kemiskinan'] = 'Cluster ' + df_f['cluster'].astype(str)

# ======================================================
# CENTROID (ATAS)
# ======================================================
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df.insert(0, 'Cluster', centroid_df.index)

if k == 3:
    centroid_df['Kategori'] = centroid_df['Cluster'].map({
        0: 'Kemiskinan Tinggi',
        1: 'Kemiskinan Sedang',
        2: 'Kemiskinan Rendah'
    })
else:
    centroid_df['Kategori'] = 'Cluster ' + centroid_df['Cluster'].astype(str)

centroid_df[features] = centroid_df[features].round(2)
centroid_df = centroid_df[['Cluster', 'Kategori'] + features]

st.subheader("🎯 Centroid Cluster")
st.caption("Rata-rata nilai tiap indikator pada masing-masing cluster")
st.dataframe(centroid_df, use_container_width=True)

st.divider()

# ======================================================
# PCA VISUAL
# ======================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

st.subheader("📌 Visualisasi Clustering (PCA)")

fig, ax = plt.subplots(figsize=(9,6))

if k == 3:
    warna = {
        'Kemiskinan Tinggi': 'red',
        'Kemiskinan Sedang': 'orange',
        'Kemiskinan Rendah': 'green'
    }
    for kategori, color in warna.items():
        idx = df_f['kategori_kemiskinan'] == kategori
        ax.scatter(X_pca[idx,0], X_pca[idx,1], c=color, label=kategori, alpha=0.6)
else:
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=df_f['cluster'], cmap='tab10', alpha=0.6)
    ax.legend(*sc.legend_elements(), title="Cluster")

ax.scatter(
    centroids_pca[:,0], centroids_pca[:,1],
    c='black', marker='X', s=220, label='Centroid'
)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)

st.divider()



# ======================================================
# DATA TABLE
# ======================================================
st.subheader("📄 Data Hasil Clustering")

st.dataframe(
    df_f[['nama_kabupaten_kota', 'tahun', 'kategori_kemiskinan'] + features],
    use_container_width=True,
    height=420
)

# ======================================================
# FOOTER
# ======================================================
st.markdown(
    """
    ---
    <div style='text-align:center; font-size:14px;'>
    📊 <b>Clustering Kemiskinan Jawa Barat</b><br>
    Metode: K-Means Clustering
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# ELBOW METHOD
# ======================================================
st.subheader("📉 Elbow Method")
st.caption("Menentukan jumlah cluster optimal")

inertia = []
K_range = range(1, 11)

for k_elbow in K_range:
    km = KMeans(n_clusters=k_elbow, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig_elbow, ax = plt.subplots(figsize=(7,5))
ax.plot(K_range, inertia, marker='o')
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Inertia")
ax.set_title("Grafik Elbow Method")
ax.grid(alpha=0.3)

st.pyplot(fig_elbow)
