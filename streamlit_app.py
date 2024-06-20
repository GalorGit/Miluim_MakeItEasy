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



# Page title
st.set_page_config(page_title='Miluim_MakeItEasy', page_icon='🤖')


filled_form = True

st.title("דף בית")
if st.button("Personal Info"):
    if filled_form:
        # Example options inside slider
        st.checkbox("My Reimbursement")
        st.checkbox("Information Left")
    else:
        st.button("Fill the Form")



if st.button("Immediate Help"):
    # Example options inside slider
    st.write("Hotlines")
    st.write("Aid Fund")
    st.write("Reserved Website")


if st.button("What Do I Deserve?"):
    # Example options inside slider
    st.write("National Insurance Institute")
    st.write("Ministry of Interior")
    st.write("Funds")











# _____________________________________________________

# Setting up the title of the app
st.title('Personal Details Form')

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

# Creating an expander for the main questionnaire
with st.expander('Main Questionnaire'):
    st.markdown('**Personal Information**')

    # Collecting basic information in Hebrew
    hebrew_first_name = st.text_input('First Name (Hebrew)').strip()
    if hebrew_first_name and not contains_only_hebrew(hebrew_first_name):
        st.error("First Name (Hebrew) should only contain Hebrew letters, spaces, or hyphens.")
    
    # hebrew_surname = st.text_input('Surname (Last Name) (Hebrew)').strip()
    # if hebrew_surname and not contains_only_hebrew(hebrew_surname):
    #     st.error("Surname (Hebrew) should only contain Hebrew letters, spaces, or hyphens.")
    
    # Collecting basic information in English
    english_first_name = st.text_input('First Name (English)').strip()
    if english_first_name and not contains_only_english(english_first_name):
        st.error("First Name (English) should only contain English letters, spaces, or hyphens.")
    
    english_surname = st.text_input('Surname (Last Name) (English)').strip()
    if english_surname and not contains_only_english(english_surname):
        st.error("Surname (English) should only contain English letters, spaces, or hyphens.")
    
    # identification = st.text_input('Identification (e.g., SSN, National ID)').strip()
    # if identification and not identification.isdigit():
    #     st.error("ID should only contain numbers.")
        
    date_of_birth = st.date_input('Date of Birth', min_value=datetime(1900, 1, 1), max_value=datetime(2025, 12, 31))
    if not is_at_least_18_years_ago(date_of_birth):
        st.error("You must be at least 18 years old.")
    
    gender = st.selectbox('Gender', ['זכר', 'נקבה'])
    marital_status = st.selectbox('Marital Status', ['רווק/ה', 'נשוי/ה', 'גרוש/ה', 'אלמן/ה'])
    nationality = st.text_input('Nationality').strip()
    birth_country = st.text_input('Birth Country:').strip()

    if birth_country.lower() != 'israel':
        year_of_aliyah = st.text_input('Year of Aliyah').strip()

    # Health Maintenance Organization (HMO) Information
    hmo_name = st.text_input('Health Maintenance Organization (HMO) Name').strip()

    # Address Information
    st.markdown('**Address**')
    street_address = st.text_input('Street Address').strip()
    city = st.text_input('City').strip()
    # state_province = st.text_input('State/Province').strip()
    postal_code = st.text_input('Postal/ZIP Code').strip()
    country = st.text_input('Country').strip()
    
    # Contact Information
    st.markdown('**Contact Information**')
    phone_number = st.text_input('Phone Number').strip()
    email_address = st.text_input('Email Address').strip()

    # Children Information
    st.markdown('**Children Information**')
    num_children = st.number_input('Number of Children', min_value=0, step=1)

    children = []
    for i in range(num_children):
        child_name_key = f'child_name_{i}'
        child_name = st.text_input(f'Full Name of Child {i+1}', key=child_name_key).strip()
        if child_name and not contains_only_hebrew(child_name):
            st.error(f"Child {i+1} Name should only contain Hebrew letters, spaces, or hyphens.")
        
        # Update the section title with the child's name or a default label
        section_title = child_name if child_name else f'Child {i+1}'
        
        st.markdown(f'**{section_title} Details**')
        child_id = st.text_input(f'ID/Identifier of {section_title}', key=f'child_id_{i}').strip()
        if child_id and not child_id.isdigit():
            st.error(f"ID/Identifier of {section_title} should only contain numbers.")
            
        child_age = st.number_input(f'Age of {section_title}', min_value=0, step=1, key=f'child_age_{i}')
        child_gender = st.selectbox(f'Gender of {section_title}', ['Male', 'Female', 'Other'], key=f'child_gender_{i}')
        child_adopted = st.checkbox(f'{section_title} is adopted', key=f'child_adopted_{i}')
        child_lives_home = st.checkbox(f'{section_title} lives at home', key=f'child_lives_home_{i}')

        # Store child details in a dictionary
        child_info = {
            'name': child_name,
            'id': child_id,
            'age': child_age,
            'gender': child_gender,
            'adopted': child_adopted,
            'lives_home': child_lives_home
        }
        children.append(child_info)

# Display the collected information (for testing purposes)
if st.button('Submit'):
    st.write('First Name (Hebrew):', hebrew_first_name)
    # st.write('Surname (Hebrew):', hebrew_surname)
    st.write('First Name (English):', english_first_name)
    st.write('Surname (English):', english_surname)
    st.write('Date of Birth:', date_of_birth)
    st.write('Gender:', gender)
    st.write('Marital Status:', marital_status)
    st.write('Nationality:', nationality)
    st.write('Birth Country:', birth_country)
    if birth_country.lower() != 'israel':
        st.write('Year of Aliyah:', year_of_aliyah)
    # st.write('Identification:', identification)
    st.write('HMO Name:', hmo_name)
    st.write('City:', city)
    st.write('Street Address:', street_address)
    # st.write('State/Province:', state_province)
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




# ********************************

with st.expander('Abou
