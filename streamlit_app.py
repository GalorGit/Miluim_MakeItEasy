import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
from datetime import datetime, date
# import pdfplumber  # Assuming using pdfplumber, adjust if using another library
import re  # For regular expressions

# Page title
st.set_page_config(page_title='Miluim_MakeItEasy', page_icon='🤖')

filled_form = True

st.title("דף בית")
if st.button("Personal Info"):
    if filled_form:
        st.checkbox("My Reimbursement")
        st.checkbox("Information Left")
    else:
        st.button("Fill the Form")

if st.button("Immediate Help"):
    st.write("Hotlines")
    st.write("Aid Fund")
    st.write("Reserved Website")

if st.button("What Do I Deserve?"):
    st.write("National Insurance Institute")
    st.write("Ministry of Interior")
    st.write("Funds")

st.title('Personal Details Form')

file = st.file_uploader("Upload 3010 Form", type=['pdf'])
if file is not None:
    st.success("File uploaded successfully.")
    try:
        df = pdf_to_dataframe(file)
        if df.empty:
            st.error("The PDF was processed, but the DataFrame is empty.")
        else:
            st.write("DataFrame successfully created from PDF:")
            st.write(df)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
else:
    st.info("Please upload a PDF file.")

# Function to read PDF and convert to DataFrame
def pdf_to_dataframe(file):
    all_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            all_text += text + "\n"
    
    lines = all_text.split('\n')

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
    max_length = max(len(lst) for lst in extracted_data.values())
    for field in fields_to_extract:
        while len(extracted_data[field]) < max_length:
            extracted_data[field].append(None)  # Fill with None or a default value

    # Creating DataFrame and adding columns
    df = pd.DataFrame(extracted_data)
    df["סה\"כ ימי מילואים"] = [total_days_miluim] * len(df)
    df["תאריך התחלה"] = [earliest_date] * len(df)
    df = df.dropna()

    return df

# Function to validate if a string contains only Hebrew letters
def contains_only_hebrew(input_string):
    return all('\u0590' <= char <= '\u05FF' or char.isspace() or char == '-' for char in input_string)

# Function to validate if a string contains only English letters
def contains_only_english(input_string):
    return all('A' <= char <= 'Z' or 'a' <= 'z' or char.isspace() or char == '-' for char in input_string)

# Function to check if a date is at least 18 years ago
def is_at_least_18_years_ago(input_date):
    today = date.today()
    return (today.year - input_date.year) > 18 or ((today.year - input_date.year) == 18 and (today.month, today.day) >= (input_date.month, input_date.day))

# Creating an expander for the main questionnaire
with st.expander('Main Questionnaire'):
    st.markdown('**Personal Information**')

    english_first_name = st.text_input('First Name (English)').strip()
    if english_first_name and not contains_only_english(english_first_name):
        st.error("First Name (English) should only contain English letters, spaces, or hyphens.")

    english_surname = st.text_input('Surname (Last Name) (English)').strip()
    if english_surname and not contains_only_english(english_surname):
        st.error("Surname (English) should only contain English letters, spaces, or hyphens.")

    date_of_birth = st.date_input('Date of Birth', min_value=datetime(1900, 1, 1), max_value=datetime(2025, 12, 31))
    if not is_at_least_18_years_ago(date_of_birth):
        st.error("You must be at least 18 years old.")

    gender = st.selectbox('Gender', ['זכר', 'נקבה'])
    marital_status = st.selectbox('Marital Status', ['רווק/ה', 'נשוי/ה', 'גרוש/ה', 'אלמן/ה'])
    nationality = st.text_input('Nationality').strip()
    birth_country = st.text_input('Birth Country:').strip()

    if birth_country.lower() != 'israel':
        year_of_aliyah = st.text_input('Year of Aliyah').strip()

    hmo_name = st.text_input('Health Maintenance Organization (HMO) Name').strip()

    st.markdown('**Address**')
    street_address = st.text_input('Street Address').strip()
    city = st.text_input('City').strip()
    postal_code = st.text_input('Postal/ZIP Code').strip()
    country = st.text_input('Country').strip()

    st.markdown('**Contact Information**')
    phone_number = st.text_input('Phone Number').strip()
    email_address = st.text_input('Email Address').strip()

    st.markdown('**Children Information**')
    num_children = st.number_input('Number of Children', min_value=0, step=1)

    children = []
    for i in range(num_children):
        child_name_key = f'child_name_{i}'
        child_name = st.text_input(f'Full Name of Child {i+1}', key=child_name_key).strip()
        if child_name and not contains_only_hebrew(child_name):
            st.error(f"Child {i+1} Name should only contain Hebrew letters, spaces, or hyphens.")

        section_title = child_name if child_name else f'Child {i+1}'

        st.markdown(f'**{section_title} Details**')
        child_id = st.text_input(f'ID/Identifier of {section_title}', key=f'child_id_{i}').strip()
        if child_id and not child_id.isdigit():
            st.error(f"ID/Identifier of {section_title} should only contain numbers.")

        child_age = st.number_input(f'Age of {section_title}', min_value=0, step=1, key=f'child_age_{i}')
        child_gender = st.selectbox(f'Gender of {section_title}', ['Male', 'Female', 'Other'], key=f'child_gender_{i}')
        child_adopted = st.checkbox(f'{section_title} is adopted', key=f'child_adopted_{i}')
        child_lives_home = st.checkbox(f'{section_title} lives at home', key=f'child_lives_home_{i}')

        child_info = {
            'name': child_name,
            'id': child_id,
            'age': child_age,
            'gender': child_gender,
            'adopted': child_adopted,
            'lives_home': child_lives_home
        }
        children.append(child_info)
#ccccc
# Display the collected information (for testing purposes)
if st.button('Submit'):
    st.write('First Name (English):', english_first_name)
    st.write('Surname (English):', english_surname)
    st.write('Date of Birth:', date_of_birth)
    st.write('Gender:', gender)
    st.write('Marital Status:', marital_status)
    st.write('Nationality:', nationality)
    st.write('Birth Country:', birth_country)
    if birth_country.lower() != 'israel':
        st.write('Year of Aliyah:', year_of_aliyah)
    st.write('HMO Name:', hmo_name)
    st.write('City:', city)
    st.write('Street Address:', street_address)
    st.write('Postal/ZIP Code:', postal_code)
    st.write('Country:', country)
    st.write('Phone Number:', phone_number)
    st.write('Email Address:', email_address)
    st.write('Number of Children:', num_children)

    for i, child in enumerate(children):
        child_label = child['name'] if child['name'] else f'Child {i+1}'
        st.write(f'{child_label} Name:', child['name'])
        st.write(f'{child_label} ID/Identifier:', child['id'])
        st.write(f'{child_label} Age:', child['age'])
        st.write(f'{child_label} Gender:', child['gender'])
        st.write(f'{child_label} Adopted:', 'Yes' if child['adopted'] else 'No')
        st.write(f'{child_label} Lives at Home:', 'Yes' if child['lives_home'] else 'No')

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1.1. Input data')
    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
