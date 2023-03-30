# -*- coding: utf-8 -*-

#Import libraries
import streamlit as st
import pandas as pd
from PIL import Image
from annotated_text import annotated_text
import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

def csv_downloader(data):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "new_text_file_{}_.csv".format(timestr)
    st.markdown("#### Download  File ####")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!!</a>'
    st.markdown(href,unsafe_allow_html=True)
    
img = Image.open('c.png')
st.set_page_config(page_title='HR Analysis', page_icon=img)

hide_menu_style = """
        <style>
        footer {visibility : hidden;}
        </style>
        """
st.markdown(hide_menu_style,unsafe_allow_html=True)
        #@MainMenu {visibility : hidden;}
#load the model from disk
import pickle
load_xg = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('train_hr.csv')

st.title('HR ANALYTICS')



def main():
    menu = ["Home","CSV","About"]
    
    choice = st.selectbox("Menu",menu)
    
    if choice == "Home":
        #st.subheader("Home")
        #st.title("Home")

        image = Image.open('ds.jpg')
        st.sidebar.info('Promotion Prediction')
        st.image(image,use_column_width=True)

        #st.write("This is an application for predicting customer churn")

        check_data = st.checkbox("View the Data")
        if check_data:
            st.write(df.head(5))
            
        #st.write("Now lets find out churn prediction with other parameters")

        st.info("Input Data")
        
        st.sidebar.header('User Input Parameters')
        
        remainder__t1__previous_year_rating = st.sidebar.selectbox('Previous Year Rating',('1.0','2.0','3.0','4.0','5.0'))
        remainder__t3__department_Operations = st.sidebar.number_input('Operations Dept',('0.0','1.0'))
        remainder__t3__department_Sales = st.sidebar.number_input("Sales Dept",('0.0','1.0'))
        remainder__t3__gender_m = st.sidebar.selectbox('Gender',('0.0','0.1'))
        remainder__t3__recruitment_channel_sourcing = st.sidebar.selectbox('Recruit Channel Sourcing',('0.0','1.0'))
        remainder__remainder__region = st.sidebar.slider('Region',min_value=1.0, max_value=34.0, value=0.0)
        remainder__remainder__age = st.number_input('Age')
        remainder__remainder__length_of_service= st.sidebar.selectbox('Length of Service',('0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0','13.0'))
        remainder__remainder__KPIs_80 = st.sidebar.selectbox('KPI>80',('0.0','1.0'))
        remainder__remainder__avg_training_score = st.sidebar.number_input('Avg Training Score')

        data = {'Previous Year Rating': remainder__t1__previous_year_rating,
                'Operations Dept': remainder__t3__department_Operations,
                'Sales Dept': remainder__t3__department_Sales,
                'Gender': remainder__t3__gender_m,
                'Recruit Channel Sourcing': remainder__t3__recruitment_channel_sourcing,
                'Region': remainder__remainder__region,
                'Age': remainder__remainder__age,
                'Length of Service': remainder__remainder__length_of_service,
                'KPI>80': remainder__remainder__KPIs_80,
                'Avg Training Score': remainder__remainder__avg_training_score}
        features_df = pd.DataFrame.from_dict([data])
       
        #st.dataframe(features_df)
        
      
        #input_df = user_input_features()
        st.subheader('User Input parameters')
        st.write(features_df)

        # Apply model to make predictions
        #preprocess_df = preprocessing(features_df)
        #prediction = load_rf.predict(preprocess_df)
        
        prediction = load_xg.predict(features_df)
        prediction_proba = load_xg.predict_proba(features_df)

        st.subheader('Predicted Output')
        st.write('Promoted' if prediction_proba[0][1] > 0.5 else 'Not Promoted')

        st.subheader('Prediction Probability')
        st.write(prediction_proba)      

        if st.sidebar.button("Promotion Prediction"):
            if prediction == 1:
                st.warning('Yes, the employee will get promoted.')
            else:
                st.success('No, the employee will not be promoted.')
        
        
    elif choice == "CSV":
        st.header('Train HR CSV')
        st.write("Shape of Dataset", df.shape)
        st.dataframe(df)
        csv_downloader(df)
    else:
        st.title("About")
        #st.subheader("About")
        st.markdown("""
         :dart:  The objective of the analysis is to predict an item when sold, 
                 what is the probability that customer would file fraudulent  / Genuine warranty 
                 and to understand important factors associated with them. \n
         :dart:  The dataset is having incidents raised by customers. Which contains an event log 
                 of an incident management process extracted from a service desk platform of an IT company.\n
         :dart:  Churn is a problem for telecom companies because it is more expensive to acquire a new customer 
                 than to keep your existing one from leaving.\n
         :dart:  Churn prediction is one of the most popular Big Data use cases in business. It consists of detecting 
                 customers who are likely to cancel a subscription to a service. \n
        """)
        st.markdown("<h3></h3>", unsafe_allow_html=True)
       
        st.subheader("Model Deployment : XGBoost Classifier")

        
if __name__=='__main__':
    main()