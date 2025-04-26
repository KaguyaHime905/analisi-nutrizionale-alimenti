import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Imposta stile
sns.set(style="whitegrid")
st.set_page_config(page_title="Analisi Nutrizionale", layout="wide")

# Titolo dell'app
st.title("Analisi dei Macronutrienti degli Alimenti")

# Caricamento del file CSV (se il file esiste nella directory corrente)
try:
    df = pd.read_csv('ingredients.csv')

    # Mostra l'anteprima del dataset
    st.subheader("Anteprima del Dataset")
    st.dataframe(df.head())

    # Selezione delle colonne dei macronutrienti
    nutrients = df[['cho_per_100g', 'protein_per_100g', 'fat_per_100g',
                    'calories_per_100g', 'fiber_per_100g']]

    # Istogrammi
    st.subheader("Distribuzione dei Macronutrienti")
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    axs = axs.flatten()
    for i, col in enumerate(nutrients.columns):
        sns.histplot(df[col], kde=True, bins=30, ax=axs[i])
        axs[i].set_title(col.replace("_", " ").capitalize())
    plt.tight_layout()
    st.pyplot(fig)

    # Heatmap di correlazione
    st.subheader("Correlazione tra macronutrienti")
    corr = nutrients.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Scatter plot calorie vs carboidrati / proteine
    st.subheader("Relazioni tra nutrienti")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x='cho_per_100g', y='calories_per_100g', ax=ax1)
        ax1.set_title("Calorie vs Carboidrati")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='protein_per_100g', y='calories_per_100g', ax=ax2)
        ax2.set_title("Calorie vs Proteine")
        st.pyplot(fig2)

    # Outlier proteine > 40
    st.subheader("Alimenti con più di 40g di proteine per 100g")
    high_protein = df[df['protein_per_100g'] > 40]
    st.dataframe(high_protein[['name', 'protein_per_100g', 'calories_per_100g', 'fat_per_100g', 'food_group']])

    # Boxplot interattivi con Plotly
    df['Vegan'] = df['is_vegan'].map({True: 'Vegan', False: 'Non Vegan'})
    st.subheader("Confronto Vegan vs Non Vegan")
    st.plotly_chart(px.box(df, x='Vegan', y='cho_per_100g',
                           title='Carboidrati per 100g: Vegan vs Non Vegan',
                           labels={'cho_per_100g': 'Carboidrati (g/100g)', 'Vegan': 'Tipo'}))

    st.plotly_chart(px.box(df, x='Vegan', y='fat_per_100g',
                           title='Grassi per 100g: Vegan vs Non Vegan',
                           labels={'fat_per_100g': 'Grassi (g/100g)', 'Vegan': 'Tipo'}))

    # Barplot gruppi alimentari
    st.subheader("🍽 Macronutrienti medi per gruppo alimentare")
    group_means = df.groupby('food_group')[['calories_per_100g', 'protein_per_100g',
                                            'fat_per_100g', 'cho_per_100g']].mean().sort_values(by='calories_per_100g', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    group_means.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Top 10 vegan per proteine
    st.subheader("Top 10 alimenti vegani per contenuto proteico")
    top_vegan = df[df['is_vegan']].sort_values(by='protein_per_100g', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_vegan, x='protein_per_100g', y='name', palette='viridis', ax=ax)
    ax.set_title("Top 10 proteine (vegani)")
    st.pyplot(fig)

    # PCA + Clustering
    st.subheader("Clustering degli alimenti (PCA + KMeans)")
    X = StandardScaler().fit_transform(nutrients)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(components)

    df['cluster'] = clusters
    df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]

    fig = px.scatter(df, x='PCA1', y='PCA2', color='cluster',
                     hover_data=['name', 'food_group', 'calories_per_100g'],
                     title='Cluster di alimenti basati sul profilo nutrizionale (PCA)',
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig)

    # Boxplot calorie per gruppo alimentare
    st.subheader("Distribuzione calorie per gruppo alimentare")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='food_group', y='calories_per_100g', ax=ax)
    plt.xticks(rotation=45)
    plt.title('Distribuzione delle calorie per gruppo alimentare')
    st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ File CSV non trovato. Assicurati che 'ingredienti.csv' sia nella stessa cartella del progetto.")

#FILTRI INTERATTIVI

# Selettore per il gruppo alimentare
food_group = st.selectbox("Seleziona il gruppo alimentare", df['food_group'].unique())

# Filtra i dati in base alla selezione
filtered_df = df[df['food_group'] == food_group]

st.subheader(f"Dataset filtrato per il gruppo alimentare: {food_group}")
st.dataframe(filtered_df.head())

# Slider per filtrare per un range di calorie
calories_range = st.slider("Seleziona il range di calorie", min_value=int(df['calories_per_100g'].min()),
                           max_value=int(df['calories_per_100g'].max()), value=(0, 100), step=1)

# Filtra i dati in base al range selezionato
filtered_calories = df[(df['calories_per_100g'] >= calories_range[0]) & (df['calories_per_100g'] <= calories_range[1])]

st.subheader(f"Alimenti con calorie tra {calories_range[0]} e {calories_range[1]}")
st.dataframe(filtered_calories.head())

# Scatter plot interattivo con Plotly
fig = px.scatter(df, x='cho_per_100g', y='calories_per_100g', color='food_group', hover_data=['name'])
st.plotly_chart(fig)

# Selettore di variabili per il grafico
selected_columns = st.multiselect("Scegli i macronutrienti da visualizzare", nutrients.columns.tolist())

# Creazione del grafico dinamico in base alla selezione
if selected_columns:
    st.subheader("Distribuzione dei Nutrienti Selezionati")
    fig, axs = plt.subplots(1, len(selected_columns), figsize=(12, 6))
    if len(selected_columns) == 1:
        axs = [axs]  # Per caso di una sola variabile, rendiamo axs una lista di un solo elemento
    for i, col in enumerate(selected_columns):
        sns.histplot(df[col], kde=True, bins=30, ax=axs[i])
        axs[i].set_title(col.replace("_", " ").capitalize())
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("🔲 Seleziona almeno una variabile per visualizzare il grafico.")
    # Selettore per il numero di cluster KMeans
n_clusters = st.slider("Numero di Cluster", min_value=2, max_value=10, value=4)

# PCA + Clustering
X = StandardScaler().fit_transform(nutrients)
pca = PCA(n_components=2)
components = pca.fit_transform(X)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(components)

df['cluster'] = clusters
df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]

# Visualizzazione del clustering
fig = px.scatter(df, x='PCA1', y='PCA2', color='cluster',
                 hover_data=['name', 'food_group', 'calories_per_100g'],
                 title=f'Cluster di alimenti basati sul profilo nutrizionale (PCA) con {n_clusters} cluster',
                 color_continuous_scale='Viridis')
st.plotly_chart(fig)