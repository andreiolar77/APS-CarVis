import csv

filename = 'target_final.csv'

# Find the length of the shortest row
with open(filename) as f:
    reader = csv.reader(f)
    shortest_row = next(reader)
    for row in reader:
        if len(row) < len(shortest_row) and len(row) > 0:
            shortest_row = row
    print(f"Shortest row: {shortest_row}")

# Truncate longer rows
with open(filename) as f_in, open('target_final_fs.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    for row in reader:
        if len(row) > len(shortest_row):
            row = row[:len(shortest_row)]
        writer.writerow(row)