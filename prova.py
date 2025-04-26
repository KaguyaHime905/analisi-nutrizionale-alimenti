import streamlit as st
import pandas as pd

st.title("Test base")

uploaded_file = st.file_uploader("Carica un CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ File caricato!")
    st.dataframe(df.head())
else:
    st.info("👆 Carica un file per iniziare.")
