import pandas as pd
import numpy as np
import os
import glob
import logging
def cleanup_old_files(directory, pattern, logger=None):
    try:
        if not os.path.exists(directory):
            return
        
        files_to_remove = glob.glob(os.path.join(directory, pattern))
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                if logger:
                    logger.info(f"Removed old file: {file_path}")
            except OSError as e:
                if logger:
                    logger.warning(f"Could not remove file {file_path}: {e}")
        
        if files_to_remove and logger:
            logger.info(f"Cleaned up {len(files_to_remove)} old files from {directory}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error cleaning up files: {e}")
        raise