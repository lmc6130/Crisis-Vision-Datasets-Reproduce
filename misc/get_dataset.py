import pandas as pd
import os
import shutil

# Read tsv
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/damage_severity/consolidated/consolidated_damage_train_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/damage_severity/consolidated/consolidated_damage_test_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/damage_severity/consolidated/consolidated_damage_dev_final.tsv', delimiter='\t')

df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_train_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_test_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/disaster_types/consolidated/consolidated_disaster_types_dev_final.tsv', delimiter='\t')

df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/humanitarian/consolidated/consolidated_hum_train_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/humanitarian/consolidated/consolidated_hum_test_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/humanitarian/consolidated/consolidated_hum_dev_final.tsv', delimiter='\t')

df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/informative/consolidated/consolidated_info_train_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/informative/consolidated/consolidated_info_test_final.tsv', delimiter='\t')
df = pd.read_csv('D:/Research/crisis_vision_benchmarks/tasks/informative/consolidated/consolidated_info_dev_final.tsv', delimiter='\t')

# Obtain label
classes = df['class_label'].unique()

# Create save directory
output_folder = 'D:/Research/crisis_vision_benchmarks/informative/dev'
for cls in classes:
    folder_path = os.path.join(output_folder, cls)
    os.makedirs(folder_path, exist_ok=True)  

for index, row in df.iterrows():
    image_path = os.path.join('D:/Research/crisis_vision_benchmarks', row['image_path'])
    class_label = row['class_label']
    destination_folder = os.path.join(output_folder, class_label)
    shutil.move(image_path, destination_folder)

print("Success!")
