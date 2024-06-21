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
    #hebrew_first_name = st.text_input('First Name (Hebrew)').strip()
    #if hebrew_first_name and not contains_only_hebrew(hebrew_first_name):
     #   st.error("First Name (Hebrew) should only contain Hebrew letters, spaces, or hyphens.")
    
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

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
      
    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')
    example_csv = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download example CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use example data**')
    example_data = st.toggle('Load example data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Initiate the model building process
if uploaded_file or example_data: 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
            
        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
    
        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]
        
        rf = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        #if 'Mse' in parameter_criterion_string:
        #    parameter_criterion_string = parameter_criterion_string.replace('Mse', 'MSE')
        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        # Convert objects to numerics
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        # Round to 3 digits
        rf_results = rf_results.round(3)
        
    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Zip dataset files
    df.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label='Download ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
                )
    
    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
    
    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
    
    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'
    
    df_prediction = pd.concat([df_train, df_test], axis=0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                        color='class'
                  )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
this is our streamlit code.
We want to save all the data that we get in this code
dont earase the code, just add what you need in order to be able to save the data into a data framw once the user clicks on submit
