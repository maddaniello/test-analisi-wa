"""
Modulo Processamento Dati
Gestisce preprocessamento, pulizia e trasformazione dei dati
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Motore di preprocessamento e pulizia dati"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def process(self, data: pd.DataFrame, 
                column_mapping: Dict, 
                params: Dict) -> pd.DataFrame:
        """Processa i dati secondo le specifiche"""
        
        # Crea una copia per evitare di modificare l'originale
        processed_data = data.copy()
        
        # Applica campionamento se necessario
        if params.get('use_sampling', False):
            sample_size = params.get('sample_size', 10000)
            if len(processed_data) > sample_size:
                processed_data = processed_data.sample(n=sample_size, random_state=42)
                logger.info(f"Dati campionati a {sample_size} righe")
        
        # Gestisci valori mancanti
        processed_data = self._handle_missing_values(processed_data, column_mapping)
        
        # Processa basandosi sui tipi di colonna
        for col, mapping_info in column_mapping.items():
            if col not in processed_data.columns:
                continue
                
            category = mapping_info.get('category', 'Altro')
            
            if category == 'Data/Ora':
                processed_data = self._process_datetime(processed_data, col)
            elif category in ['Misura Numerica', 'Valuta', 'Percentuale', 'Punteggio/Valutazione']:
                processed_data = self._process_numeric(processed_data, col)
            elif category in ['Categoria/Etichetta', 'Booleano']:
                processed_data = self._process_categorical(processed_data, col)
            elif category == 'Testo/Descrizione':
                processed_data = self._process_text(processed_data, col)
        
        # Rimuovi colonne con troppi valori mancanti
        threshold = 0.95
        missing_pct = processed_data.isnull().sum() / len(processed_data)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if len(cols_to_drop) > 0:
            processed_data = processed_data.drop(columns=cols_to_drop)
            logger.info(f"Eliminate {len(cols_to_drop)} colonne con >{threshold*100}% valori mancanti")
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Gestisce valori mancanti in modo intelligente"""
        for col in data.columns:
            if data[col].isnull().sum() == 0:
                continue
            
            missing_pct = data[col].isnull().sum() / len(data)
            
            if missing_pct > 0.5:
                # Troppi valori mancanti, considera l'eliminazione
                logger.warning(f"La colonna {col} ha {missing_pct:.1%} valori mancanti")
                continue
            
            # Imputa basandosi sul tipo di dato
            if pd.api.types.is_numeric_dtype(data[col]):
                # Usa mediana per colonne numeriche
                imputer = SimpleImputer(strategy='median')
                data[col] = imputer.fit_transform(data[[col]])
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                # Usa moda per colonne categoriche
                imputer = SimpleImputer(strategy='most_frequent')
                data[col] = imputer.fit_transform(data[[col]])
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                # Forward fill per datetime
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _process_datetime(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Estrae feature da colonne datetime"""
        try:
            # Converti a datetime se non già
            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = pd.to_datetime(data[col], errors='coerce')
            
            # Controlla se abbiamo dati datetime validi
            if data[col].notna().sum() == 0:
                logger.warning(f"Nessun valore datetime valido nella colonna {col}")
                return data
            
            # Estrai feature datetime solo se abbiamo sufficiente variazione
            unique_dates = data[col].nunique()
            if unique_dates > 1:  # Estrai feature solo se le date variano
                # Anno - solo se anni multipli
                unique_years = data[col].dt.year.nunique()
                if unique_years > 1:
                    data[f'{col}_anno'] = data[col].dt.year
                
                # Mese - solo se mesi multipli
                unique_months = data[col].dt.month.nunique()
                if unique_months > 1:
                    data[f'{col}_mese'] = data[col].dt.month
                    data[f'{col}_trimestre'] = data[col].dt.quarter
                
                # Feature giorno - solo se sufficiente variazione
                if unique_dates > 7:
                    data[f'{col}_giorno'] = data[col].dt.day
                    data[f'{col}_giorno_settimana'] = data[col].dt.dayofweek
                    data[f'{col}_è_weekend'] = data[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Giorni dall'inizio - solo se significativo
                if unique_dates > 1:
                    min_date = data[col].min()
                    days_since = (data[col] - min_date).dt.days
                    if days_since.std() > 0:  # Aggiungi solo se c'è variazione
                        data[f'{col}_giorni_da_inizio'] = days_since
            else:
                logger.info(f"La colonna {col} ha solo una data unica, salto estrazione feature")
                
        except Exception as e:
            logger.error(f"Errore processando colonna datetime {col}: {str(e)}")
        
        return data
    
    def _process_numeric(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Processa colonne numeriche"""
        try:
            # Converti a numerico se necessario
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Trasformazione log per distribuzioni asimmetriche
            if data[col].notna().sum() > 0:
                skewness = data[col].skew()
                if abs(skewness) > 1:
                    # Applica trasformazione log per valori positivi
                    if (data[col] > 0).all():
                        data[f'{col}_log'] = np.log1p(data[col])
                    
                    # Applica radice quadrata per asimmetria moderata
                    elif (data[col] >= 0).all():
                        data[f'{col}_sqrt'] = np.sqrt(data[col])
        except Exception as e:
            logger.error(f"Errore processando colonna numerica {col}: {str(e)}")
        
        return data
    
    def _process_categorical(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Processa colonne categoriche"""
        try:
            unique_values = data[col].nunique()
            
            if unique_values == 2:
                # Encoding binario per categorie binarie
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('mancante'))
                self.encoders[col] = le
            elif unique_values < 10:
                # One-hot encoding per bassa cardinalità
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
            else:
                # Target encoding o frequency encoding per alta cardinalità
                freq_encoding = data[col].value_counts() / len(data)
                data[f'{col}_freq'] = data[col].map(freq_encoding)
        except Exception as e:
            logger.error(f"Errore processando colonna categorica {col}: {str(e)}")
        
        return data
    
    def _process_text(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Processa colonne di testo"""
        try:
            # Feature di testo di base
            data[f'{col}_lunghezza'] = data[col].fillna('').str.len()
            data[f'{col}_numero_parole'] = data[col].fillna('').str.split().str.len()
            
            # Sentiment o altre feature NLP potrebbero essere aggiunte qui
        except Exception as e:
            logger.error(f"Errore processando colonna testo {col}: {str(e)}")
        
        return data
