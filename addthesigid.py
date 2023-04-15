import csv
import random

# Specify the input and output filenames
input_filename = "image_final.csv"
output_filename = "image_final_fs.csv"

# Read in the input data
with open(input_filename, newline='') as f_in:
    reader = csv.reader(f_in)
    data = [row for row in reader]

old = data[0][0]

# Generate the new top row of random integers
new_row = ["sig_id"] + [str(random.randint(0, 100)) for _ in range(len(data[0]))]

# Generate the new first column of random integers
new_column = [[str(random.randint(0, 100))] + row for row in data]

# Insert the new column at the beginning of each row
for i in range(len(data)):
    data[i] = new_column[i]

# Insert the new row at the beginning of the data
data.insert(0, new_row)

data[1][1] = old

# Write the new data to the output file
with open(output_filename, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(data)