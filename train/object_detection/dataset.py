import argparse
import csv
import fiftyone
import os
import shutil

# basic: ["Ceiling fan", "Lamp", "Light bulb", "Light switch", "Mechanical fan", "Power plugs and sockets", "Television"],
# comprehensive: ["Alarm clock", "Blender", "Coffeemaker", "Dishwasher", "Gas stove", "Microwave oven", "Oven", "Pressure cooker", "Shower", "Slow cooker", "Soap dispenser",  "Toaster", "Washing machine", "Window", "Window blind"],

splits = ['test', 'train', 'validation']
label_types = ["detections"]
classes = {"Ceiling fan", "Lamp", "Light bulb", "Light switch", "Mechanical fan", "Power plugs and sockets", "Television"}
paths = {
    'input_src': os.path.expanduser('~/') + 'fiftyone/open-images-v6',
    'input_data': os.path.expanduser('~/') + 'fiftyone/open-images-v6/{}/data',
    'input_metadata': os.path.expanduser('~/') + 'fiftyone/open-images-v6/{}/metadata/classes.csv',
    'input_labels': os.path.expanduser('~/') + 'fiftyone/open-images-v6/{}/labels/detections.csv',
    'output_data': './data',
    'output_labels': './detections.csv',
}

# Cleanup
def reset():
    if os.path.exists(paths['output_data']) and os.path.isdir(paths['output_data']):
        shutil.rmtree(paths['output_data'])

    if os.path.exists(paths['output_labels']):
        os.remove(paths['output_labels'])

# Helper Functions
def download_data():
    if os.path.exists(paths['input_src']) and os.path.isdir(paths['input_src']):
        shutil.rmtree(paths['input_src'])

    fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        splits=splits,
        label_types=label_types,
        classes=classes
    )

def copy_images(split):
    shutil.copytree(paths['input_data'].format(split), paths['output_data'], dirs_exist_ok=True)

def generate_label_map(split):
    output = {}
    with open(paths['input_metadata'].format(split)) as input_csv:
        metadata = csv.reader(input_csv)
        for row in metadata:
            output[row[0]] = row[1]

    return output

def clean_csv(split,label_map):
    input_labels = None
    with open(paths['input_labels'].format(split)) as input_csv:
        input_labels = csv.DictReader(input_csv)
        
        with open(paths['output_labels'], 'a') as output_csv:
            writer = csv.writer(output_csv)

            # Iteration
            for row in input_labels:
                img_path = paths['output_data'] + '/' + row['ImageID'] + '.jpg'
                label = label_map[row['LabelName']]
                if os.path.exists(img_path) and label in classes:
                    writer.writerow([
                        split.upper(),
                        img_path,
                        label,
                        row['XMin'],
                        row['YMin'],
                        '',
                        '',
                        row['XMax'],
                        row['YMax'],
                    ])

# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action='store_true')
    
    return parser.parse_args()

# Main
def main():
    args = parse_args()
    reset()

    # Splits
    if args.download: download_data()
    for split in splits:
        print(f'Split: {split}')
        copy_images(split)
        clean_csv(split, generate_label_map(split))

if __name__ == "__main__":
    main()