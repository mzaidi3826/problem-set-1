'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd


# Your code here
def clean_data():

    # Load the CSVs from the ETL step
    pred_universe = pd.read_csv('./data/pred_universe_raw.csv', parse_dates=['arrest_date_univ'])
    arrest_events = pd.read_csv('./data/arrest_events_raw.csv', parse_dates=['arrest_date_event'])

    # Outer join on person_id
    df_arrests = pd.merge(pred_universe, arrest_events, on = 'person_id', how = 'outer')
    
    # Create target variable y: felony arrest in 1 year after current arrest
    def was_rearrested_for_felony(row):
        person_id = row['person_id']
        arrest_date = row['arrest_date_univ']

        if pd.isnull(arrest_date):
            return 0

        one_day_later = arrest_date + pd.Timedelta(days=1)
        one_year_later = arrest_date + pd.Timedelta(days=365)

        relevant_arrests = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] >= one_day_later) &
            (arrest_events['arrest_date_event'] <= one_year_later) &
            (arrest_events['charge_degree'].str.lower() == 'felony')
        ]
        return 1 if not relevant_arrests.empty else 0

    df_arrests['y'] = df_arrests.apply(was_rearrested_for_felony, axis = 1)
    print("What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year?")
    print(df_arrests['y'].mean())

    # Create current_charge_felony: 1 if current arrest is felony
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'F').astype(int)
    print("What share of current charges are felonies?")
    print(df_arrests['current_charge_felony'].mean())

    # Create num_fel_arrests_last_year: felony arrests in year prior to current arrest
    def felony_arrests_last_year(row):
        person_id = row['person_id']
        arrest_date = row['arrest_date_univ']

        if pd.isnull(arrest_date):
            return 0

        one_year_before = arrest_date - pd.Timedelta(days = 365)
        one_day_before = arrest_date - pd.Timedelta(days = 1)

        relevant_arrests = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] >= one_year_before) &
            (arrest_events['arrest_date_event'] <= one_day_before) &
            (arrest_events['charge_degree'].str.lower() == 'felony')
        ]
        return len(relevant_arrests)

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(felony_arrests_last_year, axis = 1)
    print("What is the average number of felony arrests in the last year?")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    print("Mean of num_fel_arrests_last_year:")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    print("df_arrests.head():")
    print(df_arrests.head())

    return df_arrests
