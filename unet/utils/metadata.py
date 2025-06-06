import os
import pandas as pd
from datetime import timedelta

# Well plate data as a dictionary, structured for A1 to H12
well_data = {
    'Well ID': [f"{chr(65+i)}{j}" for i in range(8) for j in range(1, 13)],  # A1 to H12 (row-major order)
    'Treatment': [
        'media alone', 'media alone', 'media alone', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row A
        'media w/ NK cells', 'media w/ NK cells', 'media w/ NK cells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row B
        'media w/ NK cells', 'media w/ NK cells', 'media w/ NK cells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row C
        'DU145 w/ NK cells -middle', 'DU145 w/ NK cells -middle', 'DU145 w/ NK cells -middle', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row D
        'DU145 w/ NK cells', 'DU145 w/ NK cells', 'DU145 w/ NK cells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row E
        'DU145 w/ NK cells', 'DU145 w/ NK cells', 'DU145 w/ NK cells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row F
        'DU145 w/ NK cells', 'DU145 w/ NK cells', 'DU145 w/ NK cells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8',  # Row G
        'DU145 w/ NK cells -edge', 'DU145 w/ NK cells -edge', 'DU145 w/ NK cells -edge', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8'   # Row H
    ],
    'Concentration': [
        0, 0, 0, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  # Row A (10,000 ng/mL for columns 4-12)
        0, 0, 0, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,           # Row B (2,500 ng/mL for columns 4-12)
        0, 0, 0, 625, 625, 625, 625, 625, 625, 625, 625, 625,                     # Row C (625 ng/mL for columns 4-12)
        0, 0, 0, 156, 156, 156, 156, 156, 156, 156, 156, 156,                     # Row D (156 ng/mL for columns 4-12)
        0, 0, 0, 39.06, 39.06, 39.06, 39.06, 39.06, 39.06, 39.06, 39.06, 39.06,   # Row E (39.06 ng/mL for columns 4-12)
        0, 0, 0, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77,           # Row F (9.77 ng/mL for columns 4-12)
        0, 0, 0, 2.44, 2.44, 2.44, 2.44, 2.44, 2.44, 2.44, 2.44, 2.44,           # Row G (2.44 ng/mL for columns 4-12)
        0, 0, 0, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61            # Row H (0.61 ng/mL for columns 4-12)
    ],
    'Unit': ['ng/ml'] * 96, # Consistent with diagram, all concentrations in ng/ml,
    'AUC': [
        133.3826, 133.3826, 133.3826, 211.4825, 200.9065, 127.1738, 211.4825, 200.9065, 127.1738, 211.4825, 200.9065, 127.1738,  # Row A
        0, 0, 0, 125.9777, 109.4065, 67.7169, 125.9777, 109.4065, 67.7169, 125.9777, 109.4065, 67.7169,  # Row B
        0, 0, 0, 109.4074, 99.0838, 55.263, 109.4074, 99.0838, 55.263, 109.4074, 99.0838, 55.263,  # Row C
        131.9053, 131.9053, 131.9053, 100.8587, 94.3618, 50.3429, 100.8587, 94.3618, 50.3429, 100.8587, 94.3618, 50.3429,  # Row D
        0, 0, 0, 100.6643, 97.8077, 59.421, 100.6643, 97.8077, 59.421, 100.6643, 97.8077, 59.421,  # Row E
        0, 0, 0, 95.0506, 99.1353, 73.2719, 95.0506, 99.1353, 73.2719, 95.0506, 99.1353, 73.2719,  # Row F
        0, 0, 0, 102.5709, 105.029, 93.273, 102.5709, 105.029, 93.273, 102.5709, 105.029, 93.273,  # Row G
        152.0615, 152.0615, 152.0615, 172.0855, 174.3526, 148.8534, 172.0855, 174.3526, 148.8534, 172.0855, 174.3526, 148.8534  # Row H
    ],
    'SD': [
        35.3585, 35.3585, 35.3585, 19.1974, 34.0562, 14.1843, 19.1974, 34.0562, 14.1843, 19.1974, 34.0562, 14.1843,  # Row A
        0, 0, 0, 20.0135, 8.9121, 7.2875, 20.0135, 8.9121, 7.2875, 20.0135, 8.9121, 7.2875,  # Row B
        0, 0, 0, 7.0377, 4.4434, 5.2785, 7.0377, 4.4434, 5.2785, 7.0377, 4.4434, 5.2785,  # Row C
        12.6288, 12.6288, 12.6288, 3.1483, 2.1591, 8.1023, 3.1483, 2.1591, 8.1023, 3.1483, 2.1591, 8.1023,  # Row D
        0, 0, 0, 5.3256, 4.5566, 15.1483, 5.3256, 4.5566, 15.1483, 5.3256, 4.5566, 15.1483,  # Row E
        0, 0, 0, 2.7176, 2.3509, 12.0589, 2.7176, 2.3509, 12.0589, 2.7176, 2.3509, 12.0589,  # Row F
        0, 0, 0, 5.9808, 13.2224, 15.9428, 5.9808, 13.2224, 15.9428, 5.9808, 13.2224, 15.9428,  # Row G
        11.4282, 11.4282, 11.4282, 22.7351, 32.66, 17.0386, 22.7351, 32.66, 17.0386, 22.7351, 32.66, 17.0386  # Row H
    ]
}

# Create well DataFrame
well_df = pd.DataFrame(well_data)
well_df['Concentration'] = well_df.apply(
    lambda row: f"{row['Concentration']} {row['Unit']}" if row['Treatment'] not in ['media alone', 'media w/ NK cells', 'DU145 w/ NK cells', 'DU145 w/ NK cells -middle', 'DU145 w/ NK cells -edge'] else '0', 
    axis=1
)

# Directory after extracting dataset.tar
base_dir = "C:/Users/mathi/Desktop/dataset"

# Parse timestamp from filename
def parse_timestamp(filename):
    parts = filename.split('_')
    time_str = parts[1]  # e.g., '01h45m25s'
    h, m, s = map(int, [time_str.split('h')[0], 
                        time_str.split('h')[1].split('m')[0], 
                        time_str.split('m')[1].split('s')[0]])
    total_seconds = h * 3600 + m * 60 + s
    return f"{total_seconds:.0f}s"

# Collect metadata
metadata = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(('.tif', '.jpg', '.jpeg', '.png')):  # Expanded to handle multiple formats
            rel_path = os.path.relpath(os.path.join(root, file), base_dir).replace(os.sep, '/')
            image_type = 'broadband' if '1W' in file else 'fluorescent' if '1B' in file else None
            is_mask = '_GT' in file
            well_id = file.split('_')[2]  # e.g., 'D2'
            timestamp = parse_timestamp(file)

            # Match well_id with well_df data
            well_info = well_df[well_df['Well ID'] == well_id].iloc[0] if well_id in well_df['Well ID'].values else None
            treatment = well_info['Treatment'] if well_info is not None else 'unknown'
            concentration = well_info['Concentration'] if well_info is not None else 'unknown'
            auc = well_info['AUC'] if well_info is not None else 'unknown'
            sd = well_info['SD'] if well_info is not None else 'unknown'

            metadata.append({
                'filename': rel_path,
                'image_type': image_type,
                'timestamp': timestamp,
                'well_id': well_id,
                'treatment': treatment,
                'concentration': concentration,
                'auc': auc,
                'sd': sd,
                'mask': is_mask
            })

# Create DataFrame
df = pd.DataFrame(metadata)

# Save as CSV
csv_path = os.path.join(base_dir, 'metadata.csv')
df.to_csv(csv_path, index=False)
print(f"Generated {csv_path} with {len(df)} entries.")

# Save as Excel (easier readability)
excel_path = os.path.join(base_dir, 'metadata.xlsx')
df.to_excel(excel_path, index=False, engine='openpyxl')
print(f"Generated {excel_path} with {len(df)} entries.")

# Save transposed Excel for well plate view
# well_df_transposed = well_df.set_index(['Well ID']).transpose()
# transposed_excel_path = os.path.join(base_dir, 'well_plate_metadata_transposed.xlsx')
# well_df_transposed.to_excel(transposed_excel_path, index=True, engine='openpyxl')
# print(f"Generated {transposed_excel_path} for well plate view.")