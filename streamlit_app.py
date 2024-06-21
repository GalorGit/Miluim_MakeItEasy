import streamlit as st
import pandas as pd
from datetime import datetime, date

# Page title
st.set_page_config(page_title='Miluim_MakeItEasy', page_icon='🤖')

# Initialize a dictionary to store collected data
stored_data = {}

# Function to validate if a string contains only Hebrew letters
def contains_only_hebrew(input_string):
    return all('\u0590' <= char <= '\u05FF' or char.isspace() or char == '-' for char in input_string)

# Function to validate if a string contains only English letters
def contains_only_english(input_string):
    return all('A' <= char <= 'Z' or 'a' <= char <= 'z' or char.isspace() or char == '-' for char in input_string)

# Function to check if a date is at least 18 years ago
def is_at_least_18_years_ago(input_date):
    today = date.today()
    return (today.year - input_date.year) > 18 or ((today.year - input_date.year) == 18 and (today.month, today.day) >= (input_date.month, input_date.day))

# Main title and buttons
st.title("דף בית")

if st.button("Personal Info"):
    with st.expander("Personal Information"):
        english_first_name = st.text_input('First Name (English)').strip()
        english_surname = st.text_input('Surname (Last Name) (English)').strip()
        date_of_birth = st.date_input('Date of Birth', min_value=datetime(1900, 1, 1), max_value=datetime(2025, 12, 31))
        gender = st.selectbox('Gender', ['זכר', 'נקבה'])
        marital_status = st.selectbox('Marital Status', ['רווק/ה', 'נשוי/ה', 'גרוש/ה', 'אלמן/ה'])
        nationality = st.text_input('Nationality').strip()
        birth_country = st.text_input('Birth Country:').strip()

        # Update stored_data
        stored_data.update({
            'English First Name': english_first_name,
            'English Surname': english_surname,
            'Date of Birth': date_of_birth,
            'Gender': gender,
            'Marital Status': marital_status,
            'Nationality': nationality,
            'Birth Country': birth_country
        })

if st.button("Immediate Help"):
    with st.expander("Immediate Help"):
        st.write("Hotlines")
        st.write("Aid Fund")
        st.write("Reserved Website")

if st.button("What Do I Deserve?"):
    with st.expander("What Do I Deserve?"):
        st.write("National Insurance Institute")
        st.write("Ministry of Interior")
        st.write("Funds")

# Submit button to display collected data
if st.button('Submit'):
    # Display the stored_data in a DataFrame
    df = pd.DataFrame.from_dict(stored_data, orient='index', columns=['Value'])
    st.dataframe(df)

# About this app section
with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, navigate through different sections and fill in the required information.')

    st.markdown('**Under the hood**')
    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')
