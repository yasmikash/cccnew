import pandas as pd

def cough_rate_calculation():

    # Reading the data
    df = pd.read_csv('results/split0_1.csv', usecols =[0,1,2,3], header = None)
    first_frame = 0
    last_frame = 0
    counter = 0
    previous = 0
    previous_previous_index = 0
    found_cough = 0

    for index, row in df.iterrows():
        if(first_frame == 0):
            if(row[1] == 8):
                first_frame = row[0]

        if (row[1] == 8):
            if(previous != 8 and df.iloc[previous_previous_index][1] != 8):
                counter = counter + 1
            last_frame = row[0]
            found_cough = 1
        previous = row[1]
        previous_previous_index = index - 2


    cough_duration_ms = (last_frame - first_frame) * 20
    cough_duration_s = cough_duration_ms / 1000

    if(found_cough == 1):
        cough_rate_calculated = counter / cough_duration_s
    if(found_cough == 0):
        cough_rate_calculated = 0.0

    return  cough_rate_calculated
