import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Implement STEADI Fall Risk Algorithm
'''
This streamlit app is based on a free tool provided by the CDC to help you assess your risk of falling. 
https://www.cdc.gov/steadi/pdf/STEADI-Algorithm-508.pdf

(mock example using pandas and csvs but you can write back to snowflake so [this](https://medium.com/streamlit/snowflake-write-back-applications-in-5-easy-steps-with-streamlit-b0ff04b8f3c1) would make sense)
'''

DEBUG_MODE = False
# simulation mode so you don't have to click to get a nice distribution in the database for figures
SIMULATE_PATIENT_MODE = True
N_SIMULATED_PATIENTS_PER_ROUND=100

# debugging to look at the json blob to be flattened and written
VERBOSE = False

# local save location
save_location = './ignore_data/'

# csv with survey administration data (questions, reasons, scores, etc)
steadi_independent_loc = './ignore_data/steadi_independent.csv'

# make a mock database file with a csv (not going to work w/schema evolution but placeholder)
# you can write back to snowflake, see example above
database_path = f'{save_location}database4a.csv'

def ingest_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df

steadi_independent_questions = ingest_csv(steadi_independent_loc)

def initialize_pt_dict():
    pt_dict = {}
    pt_dict['survey_data'] = {}
    pt_dict['survey_data']['questions'] = steadi_independent_questions.to_dict(orient='index')
    pt_dict['demographics'] = {}
    pt_dict['survey_info'] = {}
    pt_dict['survey_info']['survey_date'] = pd.to_datetime('today')
    pt_dict['survey_info']['survey_administrator'] = 'Administrator 123'
    pt_dict['survey_info']['survey_type'] = 'long'
    pt_dict['survey_info']['survey_source'] = 'Streamlit App'
    pt_dict['survey_info']['survey_version'] = '1.0'
    pt_dict['survey_info']['survey_type'] = 'STEADI Risk Assessment Stay Independent Questionnaire'
    pt_dict['survey_info']['reference'] = 'https://www.cdc.gov/steadi/pdf/STEADI-Brochure-StayIndependent-508.pdf'
    return pt_dict

def calculate_score(pt_dict):
    yes_responses = [v['yes_score'] for k,v in pt_dict['survey_data']['questions'].items() if v['response'] == 'yes']
    no_responses = [v['no_score'] for k,v in pt_dict['survey_data']['questions'].items() if v['response'] == 'no']
    not_answered_responses = {k:v['statement'] for k,v in pt_dict['survey_data']['questions'].items() if v['response'] == 'not answered'}
    
    score = sum(yes_responses) + sum(no_responses)

    if score >= 4 or pt_dict['survey_data']['self_reported_fall_last_year']:
        level='High Fall Risk'
    elif score == 0 and len(not_answered_responses) == 0:
        level = 'Low Fall Risk'
    elif score < 4 and len(not_answered_responses) == 0:
        level = 'Low Fall Risk'
    elif score < 4 and len(not_answered_responses) > 0:
        level = 'Unknown Fall Risk (missing data)'
    else:
        raise ValueError('Something went wrong with the scoring')
    
    return score, level

def simulate_patient():
    sim_pt_dict = initialize_pt_dict()
    # make a list of the unique first names of characters in Alias
    sim_pt_dict['demographics']['first_name'] = np.random.choice(['Sydney', 'Michael', 'Marcus', 'Arvin', 'Marshall', 'Irina', 'Julian', 'Nadia', 'Katya', 'Emily', 'Eric', 'Jack'])
    # make a list of the last names of characters in Alias
    sim_pt_dict['demographics']['last_name'] = np.random.choice(['Bristow', 'Vaughn', 'Dixon', 'Sloane', 'Flinkman', 'Derevko', 'Lazarey', 'Sarkissian', 'Kaplan', 'Weiss', 'Penderecki'])
    # pick a random date between today and 60 years ago
    sim_pt_dict['demographics']['dob'] = pd.to_datetime('today') - pd.Timedelta(days=np.random.randint(365*60, 365*100))
    sim_pt_dict['demographics']['age'] = pd.to_datetime('today').year - sim_pt_dict['demographics']['dob'].year
    temp_risk_factor = (sim_pt_dict['demographics']['age']-30)/100
    for question_number in sim_pt_dict['survey_data']['questions'].keys():
        sim_pt_dict['survey_data']['questions'][question_number]['response'] = np.random.choice(
            ['yes', 'no'], p=[temp_risk_factor, 1-temp_risk_factor])
    sim_pt_dict['survey_data']['patient_notes'] = 'This is a simulated patient'
    sim_pt_dict['survey_data']['provider_notes'] = 'This is a simulated patient'
    sim_pt_dict['survey_data']['survey_timestamp'] = pd.Timestamp(datetime.now())
    sim_pt_dict['survey_data']['self_reported_fall_last_year'] = sim_pt_dict['survey_data']['questions'][0]['response'] == 'yes'
    score, level = calculate_score(sim_pt_dict)
    sim_pt_dict['survey_data']['score'] = int(score)
    sim_pt_dict['survey_data']['risk_level'] = level
    return sim_pt_dict

# simulate a patient
if SIMULATE_PATIENT_MODE:
    ctr=0
    for i in range(N_SIMULATED_PATIENTS_PER_ROUND):
        ctr+=1
        print(f'{i} i counter... {ctr}')
        pt_dict = simulate_patient()
        pt_dict['db_timestamp'] = pd.Timestamp(datetime.now())
        pt_dict['patient_id'] = abs(hash(pt_dict['demographics']['first_name'] + pt_dict['demographics']['last_name'] + pt_dict['demographics']['dob'].strftime('%Y-%m-%d')))
        pt_dict['record_id'] = abs(hash(pt_dict['demographics']['first_name'] + pt_dict['demographics']['last_name'] + pt_dict['demographics']['dob'].strftime('%Y-%m-%d') + pt_dict['db_timestamp'].strftime('%Y-%m-%d %H:%M:%S')))
        if os.path.exists(database_path):
            print('IN SIM')
            
            try:
                database_df = pd.read_csv(database_path, header=0)
                print(database_df.shape)
                print(pd.json_normalize(pt_dict).shape)
            database_df = pd.concat([database_df, pd.json_normalize(pt_dict)], ignore_index=True)
            database_df.to_csv(database_path, index=False)
        else:
            print('NO ExiSTE')
            pd.json_normalize(pt_dict).to_csv(database_path, index=False)

pt_dict = initialize_pt_dict()

st.markdown('# STEADI Fall Risk Assessment')
st.markdown('## Independent Questionnaire')
st.markdown('Answer "yes" or "no" to the following questions:')

with st.form(key='input_form'):
    pt_dict['demographics']['first_name'] = st.text_input(label='First Name:')
    pt_dict['demographics']['last_name'] = st.text_input(label='Last Name:')
    if DEBUG_MODE:
        pt_dict['demographics']['dob'] = st.date_input(
        label='Date of Birth:',
        min_value=pd.to_datetime('1900-01-01'),
        max_value=pd.to_datetime('today'),
        value=pd.to_datetime('today') - pd.Timedelta(days=np.random.randint(365*60, 365*100))
        )
    else:
        pt_dict['demographics']['dob'] = st.date_input(
            label='Date of Birth:',
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today'),
            value=pd.to_datetime('today')
        )
    pt_dict['demographics']['age'] = pd.to_datetime('today').year - pt_dict['demographics']['dob'].year
    for question_number, question_dict in pt_dict['survey_data']['questions'].items():
        st.markdown(f'### {question_dict["statement"]}')
        st.markdown(f'Why we ask: {question_dict["reason"]}')
        if DEBUG_MODE:
            temp_response = st.radio(label='Answer:', options=['yes', 'no', 'not Answered'], index=np.random.randint(2), key=f'question_{question_number}')
        else:    
            temp_response = st.radio(label='Answer:', options=['yes', 'no', 'not Answered'], index=2, key=f'question_{question_number}')
        pt_dict['survey_data']['questions'][question_number]['response'] = temp_response

    pt_dict['survey_data']['patient_notes'] = st.text_input(label='Additional notes:')
    pt_dict['survey_data']['survey_timestamp'] = pd.Timestamp(datetime.now())
    pt_dict['survey_data']['self_reported_fall_last_year'] = pt_dict['survey_data']['questions'][0]['response'] == 'yes'

    score, level = calculate_score(pt_dict)
    pt_dict['survey_data']['score'] = int(score)
    pt_dict['survey_data']['risk_level'] = level

    submit_button = st.form_submit_button(label='Show Results (not sent to database)')


def render_results(pt_dict):
    score, level = calculate_score(pt_dict)
    not_answered_responses = {k:v['statement'] for k,v in pt_dict['survey_data']['questions'].items() if v['response'] == 'not answered'}

    st.markdown('## Results')
    if len(not_answered_responses) > 0:
        st.markdown('### :orange[Unanswered Questions]')
        st.markdown('If you are able, please answer the following questions and resubmit the form. You can ask your provider for help.')
        for k,v in not_answered_responses.items():
                st.markdown(f'{k}. {v}')

    if score >= 4 or pt_dict['survey_data']['self_reported_fall_last_year']:
        level='High Fall Risk'
    elif score == 0 and len(not_answered_responses) == 0:
        level = 'Low Fall Risk'
    elif score < 4 and len(not_answered_responses) == 0:
        level = 'Low Fall Risk'
    elif score < 4 and len(not_answered_responses) > 0:
        level = 'Unknown Fall Risk (missing data)'
    else:
        raise ValueError('Something went wrong with the scoring')
    
    pt_dict['survey_data']['risk_level'] = level

    if pt_dict['demographics']['age'] < 65:
        st.markdown(f'### :orange[Invalid Age: {pt_dict["demographics"]["age"]}]')
        st.markdown(f':orange[This tool is only validated for patients 65 years and older. Was the birth date entered correctly?]')
    else:
        st.markdown(f'### Age: {pt_dict['demographics']['age']}')

    if score >= 4:
        st.markdown(f'### Score: :red[{score}/{sum([v['yes_score'] for k,v in pt_dict['survey_data']['questions'].items()])}]')
    else:
        st.markdown(f'### Score: {score}/{sum([v['yes_score'] for k,v in pt_dict['survey_data']['questions'].items()])}')

    if pt_dict['survey_data']['self_reported_fall_last_year']:
        st.markdown('### Self-reported fall in the last year: :red[Yes]')
    elif len(not_answered_responses) ==  0:
        st.markdown('### Self-reported fall in the last year: no')
    else:
        st.markdown('### Self-reported fall in the last year: Unknown (missing data)')

    #make this line red if high risk, green if low risk
    if level == 'High Fall Risk':
        st.markdown(f'### Independence Fall Risk Level: :red[{level}]')
    elif level == 'Low Risk':
        st.markdown(f'### Independence Fall Risk Level: :green[{level}]')
    else:
        st.markdown(f'### Independence Fall Risk Level: {level}')

if submit_button:
    render_results(pt_dict)   
    send_to_database = False

with st.form(key='provider_form'):
    pt_dict['survey_data']['provider_notes'] = st.text_input(label='Provider notes:')
    send_to_database = st.form_submit_button(label='Send to Database')
    if send_to_database:
        render_results(pt_dict)

def unpack_the_onion(my_dict):
    for k,v in my_dict.items():
        if type(v) == dict:
            st.markdown(f'### {k}')
            unpack_the_onion(v)
        else:
            st.markdown(f'{k}: {v}')

if send_to_database:
    pt_dict['db_timestamp'] = pd.Timestamp(datetime.now())
    pt_dict['patient_id'] = abs(hash(pt_dict['demographics']['first_name'] + pt_dict['demographics']['last_name'] + pt_dict['demographics']['dob'].strftime('%Y-%m-%d')))
    pt_dict['record_id'] = abs(hash(pt_dict['demographics']['first_name'] + pt_dict['demographics']['last_name'] + pt_dict['demographics']['dob'].strftime('%Y-%m-%d') + pt_dict['db_timestamp'].strftime('%Y-%m-%d %H:%M:%S')))

    if VERBOSE:
        unpack_the_onion(pt_dict)
    
    # flatten the dictionary
    df2db = pd.json_normalize(pt_dict)

    if os.path.exists(database_path):
        print('PATH EXISTS')
        database_df = pd.read_csv(database_path, header=0)
        database_df = pd.concat([database_df, df2db], ignore_index=True)
        database_df.to_csv(database_path, index=False)
    else:
        df2db.to_csv(database_path, index=False)
        print('ELSING')
        database_df = pd.read_csv(database_path)
    st.markdown(f'### :green[Data saved to database] Records in database: {len(database_df)}')

    database_df = database_df[database_df['demographics.age'] >= 60]
    database_df['demographics.decade'] = database_df['demographics.age'].apply(lambda x: int(x/10)*10)
    database_df['survey_data.score'] = database_df['survey_data.score'].astype(int)

    # plot distribution of scores
    fig_dist, ax_dist = plt.subplots()
    sns.countplot(data=database_df, x='survey_data.score', ax=ax_dist, color='gray')
    # label y-axis
    ax_dist.set_ylabel('# of Patients')
    

    # # plot swarm or violin plot of age vs score
    fig_scatter, ax_scatter = plt.subplots()
    risk_palette = palette={'Low Fall Risk': 'green', 'Unknown Fall Risk (missing data)':'yellow',  'High Fall Risk': 'red'}
    dark_risk_palette = palette={'Low Fall Risk': 'darkgreen', 'Unknown Fall Risk (missing data)':'darkgoldenrod',  'High Fall Risk': 'darkred'}
    sns.stripplot(data=database_df, y='demographics.decade', x='survey_data.score', orient='h',
                    hue='survey_data.risk_level', ax=ax_scatter, palette=risk_palette, dodge=False, alpha=.2, zorder=1)
    # sns.violinplot(data=database_df, y='demographics.decade', x='survey_data.score', 
    #                 hue='survey_data.risk_level', palette=risk_palette, orient='h')
    sns.pointplot(y='demographics.decade', x='survey_data.score',
              data=database_df, orient='h',
              join=False, color='black',
              markers="d", scale=.75, ci=None, label='Mean Score for Decade')
    # label y-axis
    ax_scatter.set_ylabel('Decade (Years)')
    # label x-axis
    ax_scatter.set_xlabel('Fall Risk Score')
    # set x-axis to every 1 point
    ax_scatter.set_xticks(np.arange(0, 14, 1))
    # set y-axis to every 10 years
    #ax_scatter.set_yticks(np.arange(min(database_df['demographics.decade'].apply(lambda x: int(x))), 100, 10))
    
    for ax in [ax_dist, ax_scatter]:
        # set x limits from 0 to 14
        ax.set_xlim(0, 14)
        # set x ticks from 0 to 14 by 1
        #ax.set_xticks(np.arange(-1, 14, 1))
        # get x tick labels
        xtick_label_text = [label.get_text() for label in ax.get_xticklabels()]

        pt_score_pos = xtick_label_text.index(str(pt_dict['survey_data']['score']))

        # vertical line at score
        ax.axvline(x=pt_score_pos, color='black', linestyle='--', label='Patient Score')
        # put an arrow with text "Patient Score" in yellow pointing to the axvline
        #ax.annotate('Patient Score', xy=(pt_score_pos, 2), xytext=(pt_score_pos+1, 2), color='black', arrowprops=dict(facecolor='yellow', shrink=0.05))

        ax.legend(loc='upper right')
        
        # color from 4 to xmax light red
        ax.axvspan(4, ax.get_xlim()[-1], alpha=0.3, color='red')
        # color from 0 to 4 light green
        ax.axvspan(ax.get_xlim()[0], 4, alpha=0.3, color='green')

    st.pyplot(fig_scatter)
    st.pyplot(fig_dist)

st.markdown('#### References')
st.markdown(
    'This checklist was developed by the Greater Los Angeles VA Geriatric Research Education Clinical Center'
    ' and affiliates and is a validated fall risk self-assessment tool '
    '(Rubenstein et al. J Safety Res; 2011: 42(6)493-499).'
     ' It was adapted by the CDC [in this brochure](https://www.cdc.gov/steadi/pdf/STEADI-Brochure-StayIndependent-508.pdf)'
     ' with permission of the authors.'
)