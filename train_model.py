# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

print("Inizio addestramento modello...")

# --- 1. Caricamento e Preparazione Dati ---
DATA_PATH = os.path.join('Data', 'Real estate valuation data set.xlsx')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File dataset non trovato in {DATA_PATH}. Assicurati che esista.")

df = pd.read_excel(DATA_PATH)

# Rinominiamo le colonne per chiarezza (opzionale ma utile)
# Nota: Adattalo se i nomi delle tue colonne sono leggermente diversi
# Assumiamo che le colonne siano nell'ordine corretto come da descrizione comune del dataset
column_mapping = {
    'No': 'No',
    'X1 transaction date': 'TransactionDate',
    'X2 house age': 'HouseAge',
    'X3 distance to the nearest MRT station': 'DistanceMRT',
    'X4 number of convenience stores': 'ConvenienceStores',
    'X5 latitude': 'Latitude',
    'X6 longitude': 'Longitude',
    'Y house price of unit area': 'PriceUnitArea'
}
df = df.rename(columns=column_mapping)

# Definiamo le features (X) e il target (y)
# Escludiamo 'No' e 'TransactionDate' per questo modello specifico, come da requisiti impliciti
# L'addestramento originale includeva TUTTE le colonne tranne il target. Manteniamo questo comportamento.
# Se vuoi addestrare SOLO su lat/lon, dovresti cambiare X qui.
# Ma per permettere entrambi gli input nell'app, è meglio addestrare su più features.
features = [
    'TransactionDate', # Mantenuta perché nel training originale
    'HouseAge',
    'DistanceMRT',
    'ConvenienceStores',
    'Latitude',
    'Longitude'
]
target = 'PriceUnitArea'

X = df[features]
y = df[target]

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensioni Training Set: {X_train.shape}, Test Set: {X_test.shape}")

# --- 2. Addestramento Modello ---
# Crea il modello base: un albero decisionale
base_estimator = DecisionTreeRegressor(random_state=42)

# Usa RANSACRegressor
# Nota: min_samples=0.5 significa che almeno il 50% dei dati deve essere "inlier"
ransac = RANSACRegressor(
    estimator=base_estimator,
    min_samples=0.5, # Frazione minima di campioni richiesta
    random_state=42,
    max_trials=100, # Numero massimo di iterazioni RANSAC
    stop_probability=0.99 # Soglia di probabilità per fermarsi prima
)

print("Addestramento RANSAC Regressor...")
ransac.fit(X_train, y_train)
print("Addestramento completato.")

# --- 3. Valutazione (opzionale qui, ma utile) ---
y_pred_train = ransac.predict(X_train)
y_pred_test = ransac.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"\nMean Squared Error (Train): {mse_train:.4f}")
print(f"Mean Squared Error (Test): {mse_test:.4f}")

# Inlier mask (quali campioni sono stati considerati inliers da RANSAC)
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask
print(f"Numero di inliers identificati nel training set: {sum(inlier_mask)}")
print(f"Numero di outliers identificati nel training set: {sum(outlier_mask)}")


# --- 4. Salvataggio Modello e Metadati ---

# Salva il modello addestrato
MODEL_PATH = 'ransac_model.joblib'
joblib.dump(ransac, MODEL_PATH)
print(f"Modello salvato in: {MODEL_PATH}")

# Salva informazioni utili per l'app Streamlit
# - Limiti per Latitudine e Longitudine (per validazione input)
# - Valori medi delle feature (per usarli quando l'utente fornisce solo un sottoinsieme di input)
# - Nomi delle feature nell'ordine corretto atteso dal modello
metadata = {
    'feature_names': features, # Ordine delle colonne usato per addestrare
    'latitude_min': df['Latitude'].min(),
    'latitude_max': df['Latitude'].max(),
    'longitude_min': df['Longitude'].min(),
    'longitude_max': df['Longitude'].max(),
    'feature_means': X_train.mean().to_dict() # Medie calcolate sul training set
}

METADATA_PATH = 'model_metadata.joblib'
joblib.dump(metadata, METADATA_PATH)
print(f"Metadati salvati in: {METADATA_PATH}")

print("\nScript di addestramento completato con successo.")