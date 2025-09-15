import pandas as pd
import numpy as np
import re
import unicodedata

def normalize(dataframe, logger):
    try:
        logger.info(f"Starting normalization of {len(dataframe)} records")
        
        # Regular expressions used to filter out classes of characters
        nonalpha = re.compile(r'[^A-Z]')
        nonalphanumeric = re.compile(r'[^A-Z0-9]')
        nonnumeric = re.compile(r'[^0-9]')
        # Regular expressions used to filter out last_name prefixes and suffixes
        prefix = re.compile(r'^(?:MC|MAC|DE|VAN|VON|LA|LE|DEL|DELLA|DI)')
        suffix = re.compile(r'(?:JR|SR|II|III|IV|V|JUNIOR|SENIOR)$')

        # Advanced last name cleaning with prefix/suffix removal
        if 'last_name' in dataframe.columns:
            dataframe['last_name'] = (dataframe['last_name']
                .str.normalize('NFKD')
                .str.upper()
                .str.replace(nonalpha, '', regex=True)
                .str.replace(prefix, '', regex=True)
                .str.replace(suffix, '', regex=True))
            logger.info("Normalized last_name column")

        # Middle name normalization - truncate to first character
        if 'middle_name' in dataframe.columns:
            dataframe['middle_name'] = (dataframe['middle_name']
                .str.normalize('NFKD')
                .str.upper()
                .str.replace(nonalpha, '', regex=True)
                .str[0])  # Take only first character
            logger.info("Normalized middle_name column")

        # Sex normalization - convert to uppercase and truncate to one character
        if 'sex' in dataframe.columns:
            dataframe['sex'] = (dataframe['sex']
                .str.upper()
                .str[0])
            logger.info("Normalized sex column")

        # Address, city, state normalization
        for column in ['address', 'city', 'state']:
            if column in dataframe.columns:
                dataframe[column] = (dataframe[column]
                    .str.normalize('NFKD')
                    .str.upper()
                    .str.replace(nonalphanumeric, '', regex=True))
                logger.info(f"Normalized {column} column")

        # SSN and phone normalization - keep only numeric characters
        for column in ['ssn', 'phone']:
            if column in dataframe.columns:
                dataframe[column] = (dataframe[column]
                    .str.replace(nonnumeric, '', regex=True))
                logger.info(f"Normalized {column} column")

        # ZIP code normalization - keep only numeric characters
        if 'zip' in dataframe.columns:
            dataframe['zip'] = (dataframe['zip']
                .str.replace(nonnumeric, '', regex=True))
            logger.info("Normalized zip column")

        if 'dob' in dataframe.columns:
            dataframe['dob'] = pd.to_datetime(dataframe['dob'], errors='coerce').dt.strftime('%Y-%m-%d')
            dataframe['dob'] = dataframe['dob'].replace('NaT', None)
            logger.info("Normalized dob column (kept as text for CURL compatibility)")

        # Convert all empty strings to None for consistent null handling
        dataframe.replace('', None, inplace=True)

        # Remove placeholder records
        if 'last_name' in dataframe.columns:
            initial_count = len(dataframe)
            dataframe = dataframe[~dataframe['last_name'].isin(['BABY', 'INFANT', 'VOID', 'UNKNOWN'])]
            removed_count = initial_count - len(dataframe)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} placeholder records")
        
        logger.info(f"Normalization complete: {len(dataframe)} records remaining")
        return dataframe
        
    except Exception as e:
        logger.error(f"Error in normalize function: {e}")
        raise


def main():
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description='Normalize family linkage data')
    parser.add_argument('--source-columns', required=True, help='Comma-separated list of columns')
    parser.add_argument('--input-file', required=True, help='Input CSV file')
    parser.add_argument('--output-file', required=True, help='Output file')
    parser.add_argument('--source-field-delimiter', default=',', help='Input field delimiter')
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Read input file
        columns = args.source_columns.split(',')
        df = pd.read_csv(args.input_file, delimiter=args.source_field_delimiter, 
                        usecols=columns, dtype='string')
        
        # Normalize data
        normalized_df = normalize(df, logger)
        
        # Write output
        normalized_df.to_csv(args.output_file, index=False)
        logger.info(f"Normalized data written to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
