import csv

with open('img_pixels.csv', 'r') as infile, open('image_final.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    row_count = 0
    for row in reader:
        if row_count % 2 == 0:
            writer.writerow(row)
        row_count += 1

    print(f"Processed {row_count} rows")