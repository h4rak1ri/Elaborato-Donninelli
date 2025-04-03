# Giulio Enzo Donninelli Elaborato Sistemi Informativi 03/04/2025

## Real Estate Valuation Data Set

Questo repository contiene il dataset "Real Estate Valuation Data Set", relativo al mercato immobiliare, con **414 osservazioni** e **8 variabili**. I dati offrono una panoramica delle transazioni immobiliari avvenute a Taipei, Taiwan, e includono informazioni temporali, spaziali e caratteristiche specifiche degli immobili.

## Descrizione del Dataset

Le variabili presenti sono:

- **No**: Identificativo progressivo dell’osservazione.
- **X1 transaction date**: Data della transazione (in formato numerico, es. 2013.148953).
- **X2 house age**: Età dell’immobile in anni.
- **X3 distance to the nearest MRT station**: Distanza dalla stazione della metropolitana più vicina (in metri).
- **X4 number of convenience stores**: Numero di negozi di prossimità presenti nell’area.
- **X5 latitude**: Latitudine geografica dell’immobile.
- **X6 longitude**: Longitudine geografica dell’immobile.
- **Y house price of unit area**: Prezzo per unità di area (generalmente in New Taiwan Dollar per metro quadrato).

## Analisi Descrittiva di Base

Di seguito la tabella riassuntiva degli indicatori statistici calcolati per ciascuna variabile:

| Variabile                                   | Count   | Mean         | Std         | Min        | 25%         | 50%         | 75%         | Max        |
|---------------------------------------------|---------|--------------|-------------|------------|-------------|-------------|-------------|------------|
| **X1 transaction date**                     | 414     | 2013.148953  | 0.281995    | 2012.666667| 2012.916667 | 2013.166667 | 2013.416667 | 2013.583333|
| **X2 house age**                            | 414     | 17.712560    | 11.392485   | 0.000000   | 9.025000    | 16.100000   | 28.150000   | 43.800000  |
| **X3 distance to the nearest MRT station**  | 414     | 1083.885689  | 1262.109595 | 23.382840  | 289.324800  | 492.231300  | 1454.279000 | 6488.021000|
| **X4 number of convenience stores**         | 414     | 4.094203     | 2.945562    | 0.000000   | 1.000000    | 4.000000    | 6.000000    | 10.000000  |
| **X5 latitude**                             | 414     | 24.969030    | 0.012410    | 24.932070  | 24.963000   | 24.971100   | 24.977455   | 25.014590  |
| **X6 longitude**                            | 414     | 121.533361   | 0.015347    | 121.473530 | 121.528085  | 121.538630  | 121.543305  | 121.566270 |
| **Y house price of unit area**              | 414     | 37.980193    | 13.606488   | 7.600000   | 27.700000   | 38.450000   | 46.600000   | 117.500000 |

## Distribuzione delle Variabili

### X1 transaction date
- **Descrizione:** Rappresenta il periodo temporale delle transazioni.
- **Osservazioni:** Le transazioni sono concentrate tra la fine del 2012 e la metà del 2013.
![alt text](<Img/X1 transaction date.png>)

### X2 house age
- **Descrizione:** Età degli immobili in anni.
- **Osservazioni:** Gli immobili hanno un’età media di circa 18 anni, con valori da 0 a 44 anni.
![alt text](<Img/X2 house age.png>)

### X3 distance to the nearest MRT station
- **Descrizione:** Distanza dalla stazione MRT più vicina.
- **Osservazioni:** La distanza varia notevolmente (da circa 23 a oltre 6400 metri), con una media di circa 1084 metri. La presenza di valori elevati indica immobili situati anche in zone periferiche.
![alt text](<Img/X3 distance to the nearest MRT station.png>)

### X4 number of convenience stores
- **Descrizione:** Numero di negozi di prossimità nella zona.
- **Osservazioni:** In media sono presenti circa 4 negozi per area. Alcune zone ne hanno fino a 10, altre nessuno.
![alt text](<Img/X4 number of convenience stores.png>)

### X5 latitude e X6 longitude
- **Descrizione:** Coordinate geografiche degli immobili.
- **Osservazioni:** I valori si concentrano in una specifica area urbana, rendendo possibile l’analisi spaziale.
![alt text](<Img/X5 latitude.png>)

![alt text](<Img/X6 longitude.png>)

### Y house price of unit area
- **Descrizione:** Prezzo per unità di area dell’immobile.
- **Osservazioni:** Il prezzo medio è di circa 38 unità. Alcune proprietà raggiungono oltre 100 unità, indicando un’elevata variabilità.
![alt text](<Img/Y house price of unit area.png>)

---

# Predizione Prezzo Immobili Sindian (Taiwan) con RANSAC e Streamlit

Questo progetto implementa un modello di regressione per predire il prezzo al metro quadro di immobili nella regione di Sindian, Nuova Taipei, Taiwan, utilizzando il [Real Estate Valuation Data Set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set). Include uno script per addestrare il modello e un'applicazione web interattiva costruita con Streamlit per ottenere stime.

## Scelta del Modello: RANSAC Regressor con Decision Tree

La scelta di utilizzare un **RANSAC (RANdom SAmple Consensus) Regressor** in combinazione con un **Decision Tree Regressor** come stimatore base è stata guidata dalle caratteristiche dei dati immobiliari e dagli obiettivi di robustezza del modello.

*   **Decision Tree Regressor:** Gli alberi decisionali sono potenti strumenti per modellare relazioni non lineari complesse, che sono spesso presenti nei dati immobiliari (ad esempio, l'effetto della distanza da una stazione MRT sul prezzo potrebbe non essere lineare). Sono anche relativamente interpretabili.
*   **Sensibilità agli Outlier degli Alberi:** Tuttavia, un punto debole significativo degli alberi decisionali standard è la loro **elevata sensibilità agli outlier**. Un singolo immobile con un prezzo insolitamente alto o basso (dovuto a circostanze particolari non catturate dalle feature) può influenzare pesantemente la struttura dell'albero e le sue previsioni, portando a un modello poco generalizzabile.
*   **Robustezza di RANSAC:** Qui entra in gioco RANSAC. È un algoritmo iterativo specificamente progettato per **stimare i parametri di un modello su dati che contengono outlier**. RANSAC funziona selezionando iterativamente sottoinsiemi casuali dei dati (campioni minimi), addestrando il modello base (l'albero decisionale) su questi sottoinsiemi e identificando quali punti dati del set completo sono "inliers" (cioè si adattano bene al modello stimato dal sottoinsieme) e quali sono "outliers".
*   **Sinergia:** L'algoritmo RANSAC cerca il consenso, ovvero il modello (addestrato su un sottoinsieme) che è coerente con il maggior numero possibile di punti dati (gli inliers). Il modello finale viene quindi **riaddestrato utilizzando solo gli inliers identificati**. In questo modo, **RANSAC agisce come un filtro robusto agli outlier**, permettendo all'albero decisionale sottostante di apprendere le relazioni significative dai dati "puliti" senza essere distorto dai valori anomali.

In sintesi, la combinazione RANSAC + Decision Tree mira a sfruttare la capacità dell'albero di modellare complessità non lineari, **mitigando al contempo la sua vulnerabilità agli outlier grazie al meccanismo robusto di RANSAC**, ottenendo così un modello predittivo più affidabile per dati reali potenzialmente rumorosi.

## Come Eseguire il Progetto

Segui questi passaggi per configurare ed eseguire il progetto sul tuo computer locale.

**Prerequisiti:**

*   **Python:** Assicurati di avere installato Python 3.8 o superiore. Puoi scaricarlo da [python.org](https://www.python.org/).
*   **pip:** Il gestore di pacchetti di Python (di solito incluso con Python).
*   **Git:** Necessario per clonare la repository (se stai scaricando il codice sorgente da una piattaforma come GitHub).

**Passaggi:**

1.  **Clona la Repository (se necessario):**
    Se hai scaricato un file ZIP, decomprimilo. Se stai usando Git, apri il terminale o prompt dei comandi e clona la repository:
    ```bash
    git clone <URL_DELLA_TUA_REPOSITORY>
    cd <NOME_DELLA_CARTELLA_PROGETTO>
    ```
    Sostituisci `<URL_DELLA_TUA_REPOSITORY>` con l'URL effettivo e `<NOME_DELLA_CARTELLA_PROGETTO>` con il nome della cartella creata.

2.  **Crea un Ambiente Virtuale (Consigliato):**
    È buona pratica isolare le dipendenze del progetto. Esegui questi comandi nella cartella principale del progetto:
    ```bash
    # Crea l'ambiente virtuale (potrebbe essere python3 invece di python)
    python -m venv venv

    # Attiva l'ambiente virtuale
    # Su Windows:
    .\venv\Scripts\activate
    # Su macOS/Linux:
    source venv/bin/activate
    ```
    Vedrai `(venv)` all'inizio della riga del terminale, indicando che l'ambiente è attivo.

3.  **Installa le Librerie Necessarie:**
    Le dipendenze sono elencate nel file `requirements.txt` (se presente) o possono essere installate direttamente. Assicurati che il tuo ambiente virtuale sia attivo ed esegui:
    ```bash
    pip install pandas openpyxl scikit-learn streamlit joblib
    ```
    *   `pandas`: Per la manipolazione dei dati.
    *   `openpyxl`: Necessario a `pandas` per leggere file `.xlsx`.
    *   `scikit-learn`: Per il modello di machine learning (RANSAC, DecisionTree).
    *   `streamlit`: Per creare l'applicazione web interattiva.
    *   `joblib`: Per salvare e caricare il modello addestrato.

4.  **Assicurati che i Dati Siano Presenti:**
    Verifica che il file `Real estate valuation data set.xlsx` si trovi nella sottocartella `Data/` all'interno della cartella principale del progetto.

5.  **Addestra il Modello:**
    Prima di poter eseguire l'app web, devi addestrare il modello. Esegui lo script di addestramento:
    ```bash
    python train_model.py
    ```
    Questo script caricherà i dati, addestrerà il modello RANSAC e salverà due file nella cartella principale:
    *   `ransac_model.joblib`: Il modello addestrato.
    *   `model_metadata.joblib`: Metadati utili per l'app (limiti delle feature, medie, ecc.).

6.  **Avvia l'Applicazione Web Streamlit:**
    Una volta addestrato il modello, puoi lanciare l'interfaccia web:
    ```bash
    streamlit run app.py
    ```
    Questo comando avvierà un server locale e aprirà automaticamente l'applicazione nel tuo browser web predefinito. Potrai quindi interagire con l'interfaccia per ottenere le stime dei prezzi.

7.  **Interrompere l'Applicazione:**
    Per fermare l'applicazione Streamlit, torna al terminale dove hai eseguito il comando `streamlit run` e premi `Ctrl + C`.


## Mappa interattiva:
[Open the link](https://public.tableau.com/views/ElaboratoDonninelli/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)