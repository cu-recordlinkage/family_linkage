import pandas as pd
import numpy as np
import re
import unicodedata

def normalize(dataframe, logger):
    try:
        df = dataframe.copy()
        logger.info(f"Starting normalization of {len(df)} records")
        text_columns = ['last_name', 'middle_name', 'address', 'city', 'state']
        
        for col in text_columns:
            if col in df.columns:
                if col == 'last_name':
                    df[col] = df[col].apply(_clean_name_advanced)
                else:
                    df[col] = df[col].apply(_normalize_string_unified)
                logger.info(f"Normalized column: {col}")
        
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
            logger.info("Converted DOB to datetime")
        
        if 'ssn' in df.columns:
            df['ssn'] = df['ssn'].replace('--', np.nan)
            logger.info("Cleaned SSN column")
        
        if 'last_name' in df.columns:
            keywords = ['BABY', 'INFANT', 'VOID', 'UNKNOWN']
            pattern = f"^({'|'.join(keywords)})"
            placeholder_mask = df['last_name'].str.match(pattern, na=False)
            df = df[~placeholder_mask]
            logger.info(f"Removed {placeholder_mask.sum()} placeholder records")
        
        logger.info(f"Normalization complete: {len(df)} records remaining")
        return df
        
    except Exception as e:
        logger.error(f"Error in normalize function: {e}")
        raise

def _normalize_string_unified(text):
    try:
        if pd.isna(text) or not isinstance(text, str):
            return None
        
        # Strip and convert to uppercase
        text = text.strip().upper()
        
        # Normalize Unicode and remove diacritical marks
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Remove hyphens and collapse spaces
        text = re.sub(r'[-\s]+', '', text)
        
        # Keep only alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text if text else None
    except Exception:
        return text

def _clean_name_advanced(name):
    try:
        if pd.isna(name) or not isinstance(name, str):
            return None
        
        name = _normalize_string_unified(name)
        if not name:
            return None
        
        suffixes = ['JR', 'SR', 'II', 'III', 'IV', 'V', 'JUNIOR', 'SENIOR']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        prefixes = ['MC', 'MAC', 'DE', 'VAN', 'VON', 'LA', 'LE', 'DEL', 'DELLA', 'DI']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        name = re.sub(r'[^A-Z]', '', name)
        
        return name if name else None
    except Exception as e:
        return name
