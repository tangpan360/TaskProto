import os
import json
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))

def convert_gaia_templates():
    input_path = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'drain_models', 'gaia_templates.csv')
    output_path = os.path.join(_eadro_root, 'data', 'parsed_data', 'GAIA', 'templates.json')
    
    df = pd.read_csv(input_path)
    templates = sorted(df['template'].tolist())
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(templates, f)

if __name__ == '__main__':
    convert_gaia_templates()
