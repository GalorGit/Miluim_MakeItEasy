

import fitz  # PyMuPDF
import pandas as pd
import re

def pdf_to_text(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    all_text = ""

    # Iterate through each page
    for page_num in range(document.page_count):
        page = document[page_num]
        # Extract text from the page
        text = page.get_text("text")
        all_text += text + "\n"

    return all_text

# Example usage
pdf_path = "/Users/rolllab/Downloads/Form3010.pdf"
extracted_text = pdf_to_text(pdf_path)

# Split the text into lines
lines = extracted_text.split('\n')

# Initialize variables
start_line = None
end_line = None
dates = []
total_days_miluim = 0

# Define date pattern for matching dates (dd/mm/yyyy)
date_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')

# Find the start and end lines for processing
for i, line in enumerate(lines):
    if "סה\"כ ימים" in line:
        start_line = i  # Start 5 lines after "סה"כ ימים"
    if ".. אישור זה נכון ליום הוצאתו2" in line:
        end_line = i - 2  # End 4 lines before ".. אישור זה נכון ליום הוצאתו2"
        break

# Extract dates and sum numeric values
if start_line is not None and end_line is not None:
    for i in range(start_line, end_line):
        line = lines[i].strip()
        if date_pattern.match(line):  # Check if the line contains a date
            dates.append(line)
        else:
            try:
                total_days_miluim += float(line)
            except ValueError:
                continue

# Parse dates to consider the year and find the earliest date
def parse_date(date_str):
    # Convert date string from dd/mm/yyyy to (yyyy, mm, dd) for comparison
    day, month, year = map(int, date_str.split('/'))
    return (year, month, day)

# Find the earliest date by sorting based on parsed dates
if dates:
    earliest_date = min(dates, key=parse_date)
else:
    earliest_date = "Date not found"

# Fields to extract
fields_to_extract = ["מספר אישי", "שם משפחה", "שם פרטי", "תעודת זהות"]

# Initialize a dictionary to store the extracted data
extracted_data = {field: [] for field in fields_to_extract}

# Extract fields and their values
for i, line in enumerate(lines):
    line = line.strip()
    for field in fields_to_extract:
        if field in line:
            if i > 0:  # Ensure there's a previous line to look at
                value_line = lines[i - 1].strip()
                extracted_data[field].append(value_line)

# Ensure all lists in extracted_data have the same length
# Find the maximum length of the lists
max_length = max(len(lst) for lst in extracted_data.values())

# Adjust lengths to match the longest list
for field in fields_to_extract:
    while len(extracted_data[field]) < max_length:
        extracted_data[field].append(None)  # Fill with None or a default value

# Adding new columns without adding unnecessary rows
# Ensure we only add as many rows as there are in the existing DataFrame


df = pd.DataFrame(extracted_data)
    
df["סה\"כ ימי מילואים"] = [total_days_miluim] * len(df)
df["תאריך התחלה"] = [earliest_date] * len(df)
df = df.dropna()

# Output the DataFrame
df



