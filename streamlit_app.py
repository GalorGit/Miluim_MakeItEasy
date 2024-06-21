import streamlit as st
import pandas as pd
import altair as alt
import time
import zipfile
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
# n


# Setting up the title of the app
st.title('Personal Details Form')

file = st.file_uploader("Upload PDF", type=['pdf'])
if file is not None:
    st.success("File uploaded successfully.")
    # You can process the uploaded file here, such as saving it or displaying its contents
else:
    st.info("Please upload a PDF file.")

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

# Initialize a dictionary to store collected data
stored_data = {}

# Main Streamlit app structure
st.title("דף בית")

# Sidebar for navigation and data upload
with st.sidebar:
    st.title('Navigation')
    st.subheader('Select an Option')
    if st.button("Personal Info"):
        st.checkbox("My Reimbursement")
        st.checkbox("Information Left")
    if st.button("Immediate Help"):
        st.write("Hotlines")
        st.write("Aid Fund")
        st.write("Reserved Website")
    if st.button("What Do I Deserve?"):
        st.write("National Insurance Institute")
        st.write("Ministry of Interior")
        st.write("Funds")

    # About this app section
    with st.expander('About this app'):
        st.markdown('**What can this app do?**')
        st.info('This app allows users to interactively collect personal information and build a machine learning model in an end-to-end workflow.')

        st.markdown('**How to use the app?**')
        st.warning('To engage with the app, fill in personal information in the main section, then upload a CSV file or use the example data to build and evaluate a machine learning model.')

        st.markdown('**Under the hood**')
        st.markdown('Libraries used:')
        st.code('''- Pandas for data manipulation
- Scikit-learn for machine learning
- Altair for visualizations
- Streamlit for creating the web interface''', language='markdown')

# Main questionnaire expander for collecting personal information
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

# Submit button to display collected data
if st.button('Submit'):
    stored_data.update({
        'English First Name': english_first_name,
        'English Surname': english_surname,
        'Date of Birth': date_of_birth,
        'Gender': gender,
        'Marital Status': marital_status,
        'Nationality': nationality,
        'Birth Country': birth_country,
        'HMO Name': hmo_name,
        'Street Address': street_address,
        'City': city,
        'Postal/ZIP Code': postal_code,
        'Country': country,
        'Phone Number': phone_number,
        'Email Address': email_address,
        'Number of Children': num_children,
        'Children Information': children
    })

    # Display the collected information
    st.write('Collected Information:')
    st.write(stored_data)

# Model building section
with st.expander('Model Building'):
    # Upload and preprocess data
    st.header('Upload and Preprocess Data')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Uploaded Data:')
        st.write(df.head())

    else:
        st.info('Please upload a CSV file.')

    # Model parameters and training
    if st.button('Train Model'):
        if uploaded_file is not None:
            st.write('Training Model...')

            # Example model training process
            X = df.drop(columns=['target_column'])
            y = df['target_column']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R-squared: {r2}')

            # Example feature importance visualization
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            sorted_importances = feature_importances.sort_values(ascending=False)[:10]

            st.write('Feature Importances:')
            st.write(sorted_importances)

            st.write('Feature Importance Plot:')
            chart = alt.Chart(sorted_importances.reset_index()).mark_bar().encode(
                x='index',
                y='value'
            )
            st.altair_chart(chart, use_container_width=True)

            # Example model evaluation and download
            st.write('Model Evaluation and Download:')

           
