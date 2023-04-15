import os
import json
import csv

factor = 8 

# Path to the folder containing the JSON files
folder_path = "C:/Users/andre/OneDriveDocuments\APS360\dataset\VIL100\JPEGImages\0_Road001_Trim003_frames"

# Output CSV file path
output_file = "points.csv"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        # Load the JSON file
        with open(os.path.join(folder_path, filename), "r") as f:
            data = json.load(f)

        # Extract the x and y coordinates from the "points" field
        x_coords = []
        y_coords = []
        for point in data["annotations"]["lane"][0]["points"]:
            x_coords.append((point[0]/factor))
            y_coords.append((point[1]/factor))

        # Write the x and y coordinates as a row in the output CSV file
        with open(output_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(x_coords + y_coords)

