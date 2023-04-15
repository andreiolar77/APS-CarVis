import csv

#for images, rows were 2073600 in length, 400 rows including blanks
#for targets, rows ranged from 62 to 148, 400 rows including blanks
filename = 'target_final_fs.csv'

with open(filename) as f:
    reader = csv.reader(f)
    num_rows = sum(1 for row in reader)
    
with open(filename) as f:
    reader = csv.reader(f)
    shortest = float('inf')
    longest = 0
    for row in reader:
        if len(row) > 0:
            shortest = min(shortest, len(row))
            longest = max(longest, len(row))

print(f"Number of rows: {num_rows}")
print(f"Shortest row length greater than zero: {shortest}")
print(f"Longest row length greater than zero: {longest}")