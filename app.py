# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- Costanti e Caricamento Modello/Metadati ---
MODEL_PATH = 'ransac_model.joblib'
METADATA_PATH = 'model_metadata.joblib'

# THIS LINE MUST BE FIRST!
st.set_page_config(page_title="Predizione Prezzo Immobili Sindian", layout="wide")


# Funzione per caricare modello e metadati (con cache per performance)
@st.cache_resource # Usa cache_resource per oggetti non serializzabili come i modelli
def load_model_and_metadata():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        st.error(f"Errore: File modello '{MODEL_PATH}' o metadati '{METADATA_PATH}' non trovati.")
        st.error("Assicurati di aver eseguito prima lo script 'train_model.py'.")
        st.stop() # Interrompe l'esecuzione dell'app se i file mancano
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        return model, metadata
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello o dei metadati: {e}")
        st.stop()

model, metadata = load_model_and_metadata()
feature_names = metadata['feature_names']
feature_means = metadata['feature_means']
lat_min, lat_max = metadata['latitude_min'], metadata['latitude_max']
lon_min, lon_max = metadata['longitude_min'], metadata['longitude_max']

# --- Interfaccia Utente Streamlit ---

st.title("🏠 Predizione Prezzo Immobili (Sindian, Taiwan)")
st.write("""
Questa applicazione permette di stimare il prezzo al metro quadro di un immobile
nella regione di Sindian, Nuova Taipei, Taiwan, utilizzando un modello di regressione.
Puoi inserire le coordinate geografiche oppure altre caratteristiche dell'immobile.
""")

# --- Selezione Metodo di Input ---
st.sidebar.header("Scegli il Metodo di Input")
input_method = st.sidebar.radio(
    "Come vuoi fornire i dati?",
    ('Coordinate Geografiche', 'Altre Caratteristiche')
)

st.sidebar.markdown("---") # Separatore

# --- Input Utente ---
input_data = {}
predict_button = False # Inizializza il bottone

if input_method == 'Coordinate Geografiche':
    st.sidebar.subheader("Inserisci Coordinate")
    lat = st.sidebar.number_input(
        f"Latitudine ({lat_min:.4f} - {lat_max:.4f})",
        min_value=lat_min,
        max_value=lat_max,
        value=(lat_min + lat_max) / 2, # Valore di default (centro range)
        step=0.0001,
        format="%.4f",
        help=f"Inserisci la latitudine. Deve essere compresa tra {lat_min:.4f} e {lat_max:.4f}."
    )
    lon = st.sidebar.number_input(
        f"Longitudine ({lon_min:.4f} - {lon_max:.4f})",
        min_value=lon_min,
        max_value=lon_max,
        value=(lon_min + lon_max) / 2, # Valore di default (centro range)
        step=0.0001,
        format="%.4f",
        help=f"Inserisci la longitudine. Deve essere compresa tra {lon_min:.4f} e {lon_max:.4f}."
    )

    # Validazione (anche se number_input la fa già, aggiungiamo un controllo esplicito per sicurezza)
    valid_lat = lat_min <= lat <= lat_max
    valid_lon = lon_min <= lon <= lon_max

    if not valid_lat or not valid_lon:
        st.sidebar.error("Valori di latitudine o longitudine fuori dai limiti del dataset.")
    else:
        # Prepara i dati per la predizione usando le medie per le altre features
        input_data = feature_means.copy() # Inizia con le medie
        input_data['Latitude'] = lat
        input_data['Longitude'] = lon
        predict_button = st.sidebar.button("📈 Stima Prezzo (per Coordinate)", key="coord_predict")


elif input_method == 'Altre Caratteristiche':
    st.sidebar.subheader("Inserisci Caratteristiche Immobile")
    age = st.sidebar.number_input(
        "Età dell'immobile (anni)",
        min_value=0.0,
        max_value=100.0, # Limite ragionevole
        value=float(feature_means.get('HouseAge', 10.0)), # Usa media o default
        step=0.1,
        format="%.1f"
    )
    mrt_dist = st.sidebar.number_input(
        "Distanza dalla stazione MRT più vicina (metri)",
        min_value=0.0,
        max_value=10000.0, # Limite ragionevole
        value=float(feature_means.get('DistanceMRT', 1000.0)),
        step=10.0,
        format="%.1f"
    )
    stores = st.sidebar.number_input(
        "Numero di minimarket nelle vicinanze",
        min_value=0,
        max_value=20, # Limite ragionevole
        value=int(feature_means.get('ConvenienceStores', 5)),
        step=1
    )

    # Prepara i dati per la predizione usando le medie per lat/lon e transaction date
    input_data = feature_means.copy() # Inizia con le medie
    input_data['HouseAge'] = age
    input_data['DistanceMRT'] = mrt_dist
    input_data['ConvenienceStores'] = stores
    predict_button = st.sidebar.button("📈 Stima Prezzo (per Caratteristiche)", key="feat_predict")


# --- Predizione e Visualizzazione Risultato ---
st.markdown("---") # Separatore nel pannello principale
st.subheader("Risultato della Predizione")

if predict_button and input_data:
    try:
        # Crea un DataFrame con una sola riga, assicurando l'ordine corretto delle colonne
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names] # Riordina le colonne come nel training

        # Esegui la predizione
        prediction = model.predict(input_df)
        predicted_price = prediction[0] # Prendi il primo (e unico) risultato

        st.metric(
            label="Prezzo Stimato al Metro Quadro (TWD)",
            value=f"{predicted_price:.2f}"
        )

        # Mostra i dati usati per la predizione (per trasparenza)
        st.write("Dati utilizzati per la stima:")
        # Formatta l'output per una migliore leggibilità
        display_data = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in input_data.items()}
        st.json(display_data) # Mostra come JSON o usa st.dataframe(input_df)

    except Exception as e:
        st.error(f"Si è verificato un errore durante la predizione: {e}")
        st.error("Dati di input problematici:")
        st.json(input_data)

elif not predict_button:
    st.info("Inserisci i dati nella sidebar a sinistra e clicca il pulsante 'Stima Prezzo' per ottenere un risultato.")

# --- Informazioni Aggiuntive (Opzionale) ---
st.markdown("---")
with st.expander("Dettagli Tecnici"):
    st.write(f"**Modello Utilizzato:** RANSAC Regressor con base DecisionTreeRegressor")
    st.write(f"**Features usate per l'addestramento:** {', '.join(feature_names)}")
    st.write(f"**Range Latitudine Dataset:** {lat_min:.4f} - {lat_max:.4f}")
    st.write(f"**Range Longitudine Dataset:** {lon_min:.4f} - {lon_max:.4f}")
    st.write(f"**Nota:** Quando si forniscono solo alcune features, le altre vengono impostate ai loro valori medi calcolati sul dataset di addestramento.")
    st.write("Valori medi usati per il riempimento:")
    st.json({k: f"{v:.2f}" for k, v in feature_means.items()}) # Mostra medie formattate