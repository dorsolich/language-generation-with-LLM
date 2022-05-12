from question_generation.config.config import get_logger, DATA_DIR
import os
from zipfile import ZipFile
import json

_logger = get_logger(logger_name=__name__)

_logger.info(f"cwd: {os.getcwd()}")
learningQ = DATA_DIR / 'LearningQ.zip'
cleaned_learningQ = DATA_DIR / "LearningQdata.json"

# accesing the .zip file
with ZipFile(learningQ) as zf:
    
    # screening each path, and selecting the path with my criteria (where the .txt files are) in the variable paths
    paths = [path for path in zf.namelist() if 'data/experiments' in path and path.endswith('.txt') and 'MACOSX' not in path]
    
    # accesing the data whithin each path
    # storing data in a suitable data structure
    data = {}
    for path in paths:
        source = 't-' if 'teded' in path else 'k-'
        file = path.split('/')[-1]
        file = file[:file.index('.')]
        file = source+file if file.endswith('test') else file
        with zf.open(path) as f:
            data[file] = f.read().decode().split('\n')
            
for key in data.keys():
    data[key].pop(-1)
    print('len ', key, len(data[key]))
    
# saving the clean database...
with open(cleaned_learningQ, 'w') as outfile:
    json.dump(data, outfile)
    _logger.info(f"Data saved in: {cleaned_learningQ}")