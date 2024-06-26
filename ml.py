import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# 1. Load the pre-trained model
with open('model.pickle', 'rb') as model_file:
    bagging_model = pickle.load(model_file)

with open('normalisasi.pkl', 'rb') as file:
    MinMaxScaler = pickle.load(file)

# Fungsi utama Streamlit
def main():
    st.title("Klasifikasi Abalone")

    # Sidebar untuk pengguna memasukkan data
    st.sidebar.header("Masukkan Data")
    input_data = []
    st.sidebar.markdown("Length")
    input_Length = st.sidebar.number_input(f"Length", value=0.000, key = "Length")

    st.sidebar.markdown("Diameter")
    input_Diameter = st.sidebar.number_input(f"Diameter", value=0.000, key = "Diameter")

    st.sidebar.markdown("Height")
    input_Height = st.sidebar.number_input(f"Height", value=0.000, key = "Height")

    st.sidebar.markdown("Whole_Weight")
    input_Whole_Weight = st.sidebar.number_input(f"Whole_Weight", value=0.000, key = "Whole_Weight")

    st.sidebar.markdown("Shucked_weight")
    input_Shucked_weight = st.sidebar.number_input(f"Shucked_weight", value=0.000, key = "Shucked_weight")

    st.sidebar.markdown("Viscera_weight")
    input_Viscera_weight = st.sidebar.number_input(f"Viscera_weight", value=0.000, key = "Viscera_weight")

    st.sidebar.markdown("Shell_weight")
    input_Shell_weight = st.sidebar.number_input(f"Shell_weight", value=0.000, key = "Shell_weight")

    st.sidebar.markdown("Rings")
    input_Rings = st.sidebar.number_input(f"Rings", value=0.000, key = "Rings")

    # Menampilkan data yang dimasukkan pengguna
    input_dataset = [[input_Length,input_Diameter,input_Height,input_Whole_Weight,input_Shucked_weight ,input_Viscera_weight,input_Shell_weight,input_Rings]]

    input_df = pd.DataFrame({"Length": [input_Length],"Diameter": [input_Diameter],"Height": [input_Height],"Whole_Weight": [input_Whole_Weight],"Shucked_weight": [input_Shucked_weight],"Viscera_weight":[input_Viscera_weight],"Shell_weight":[input_Shell_weight],"Rings": [input_Rings]})
    st.subheader("Data yang Dimasukkan")
    st.write(input_df)

    # Melatih model jika tombol ditekan
    if st.sidebar.button("Train Model"):

        # Latih model
        data_normal = MinMaxScaler.transform(input_df)

        # Lakukan prediksi
        prediction = bagging_model.predict(data_normal)

        st.subheader("Hasil Prediksi")
        if prediction == 0:
            st.write("Prediksi Kelas: M")
        elif prediction == 1:
            st.write("Prediksi Kelas: F")
        elif prediction == 2:
            st.write("Prediksi Kelas: I")
        
        st.write()

if __name__ == "__main__":
    main()

