# Analisi Nutrizionale degli Alimenti

Questo progetto fornisce un'analisi approfondita di un dataset di alimenti, focalizzandosi sui macronutrienti (carboidrati, proteine, grassi, fibre e calorie) attraverso analisi statistiche, visualizzazioni, PCA e clustering.  
Inoltre è stata realizzata una **Web App interattiva** tramite **Streamlit**.

---

## 📦 Contenuto

- **Data Analysis Notebook** (`analisi_dati.py`): script per l'analisi esplorativa del dataset `ingredients.csv`.
- **Streamlit App** (`app.py`): app web interattiva per esplorare il dataset.
- **Dataset** (`ingredients.csv`): file CSV contenente i dati degli alimenti.

---

## 📈 Funzionalità principali

- Distribuzioni dei macronutrienti tramite istogrammi e KDE
- Matrice di correlazione tra i nutrienti
- Scatter plot delle relazioni tra calorie e nutrienti
- Analisi outlier ad alto contenuto proteico
- Confronto tra alimenti Vegan e Non Vegan
- Barplot dei macronutrienti medi per gruppo alimentare
- Classifica dei Top 10 alimenti vegani più proteici
- PCA + Clustering con KMeans
- Filtri interattivi per:
  - Gruppo alimentare
  - Range di calorie
  - Selezione dei macronutrienti da visualizzare
  - Numero di cluster KMeans dinamico

---

## 🛠 Librerie utilizzate

- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `streamlit`

---