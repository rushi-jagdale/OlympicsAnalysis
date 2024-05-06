import streamlit as st
# from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import data_preprocessing
import olympic_medal_analysis
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import pickle
# from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Olympic",
    page_icon='oly.png',
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("data_transformer.pkl", "rb") as f:
        data_pipeline = pickle.load(f)

with open("logistic_regression_model.pkl", "rb") as f:
    logistic_regression = pickle.load(f)

with open("randomforest.pkl", "rb") as f:
    random_forest = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    decision_tree = pickle.load(f)

with open("linear_regression_model.pkl", "rb") as f:
    linear_regression = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

    
with open("gradient_boost.pkl", "rb") as f:
    gb = pickle.load(f)
# random_forest = pickle.load(open("random_forest_classifier_model.pkl","rb"))

# ann_model = load_model("ann_model.h5")

def olympics_medal_predictor():
    # Load models and transformer    

    # Define available options
    sport_options = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon', 'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping', 'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']
    country_options = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA', 'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']

    # Medal Predictor
    st.title("Olympics Medal Predictor")

    # User input widgets
    with st.form("my_form"):
        st.header("Enter Athlete Details")
        Sex = st.selectbox("Sex", ["M", "F"])
        Age = st.slider("Age", min_value=10, max_value=97, step=1)
        Height = st.slider("Height (cm)", min_value=127, max_value=226, step=1)
        Weight = st.slider("Weight (kg)", min_value=25, max_value=214, step=1)
        region = st.selectbox("Country", country_options)
        Sport = st.selectbox("Sport", sport_options)
        input_model = st.selectbox("Prediction Model", ["Random Forest Classifier", "Logistic Regression", "Decision Tree", "KNN","GB" ])
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data
        inputs = pd.DataFrame([[Sex, region, Sport, Height, Weight, Age]], columns=["Sex", "region", "Sport", "Height", "Weight", "Age"])

        # Transform input data
        inputs_transformed = data_pipeline.transform(inputs)

        # Select prediction model
        if input_model == "Random Forest Classifier":
            model = random_forest
        elif input_model == "Logistic Regression":
            model = logistic_regression
        elif input_model == "Decision Tree":
            model = decision_tree
        elif input_model == "GB":
            model = gb
        elif input_model == "KNN":
            model = knn

        # elif input_model == "Neural Network":
        #     model = ann_model

        # Make prediction
        prediction = model.predict(inputs_transformed)
        # probability = model.predict_proba(inputs_transformed)[:, 1]

        # Display prediction result
        st.subheader("Prediction Result")
        # Make prediction
        prediction = model.predict(inputs_transformed)
        probability = model.predict_proba(inputs_transformed)[:, 1]  # Probability of winning (class 1)
        win_percentage = probability[0] * 100

        if win_percentage < 40:
            win_label = "Low"
        elif win_percentage >= 40 and win_percentage < 70:
            win_label = "Moderate"
        else:
            win_label = "High"

        if prediction[0] == 0:
            st.warning("Medal winning probability is {} ({}%)".format(win_label, round(win_percentage, 2)), icon="⚠️")
        else:
            st.success("Medal winning probability is {} ({}%)".format(win_label, round(win_percentage, 2)), icon="✅")
# Load and preprocess data
df = data_preprocessing.data_process()

# Sidebar radio button to select analysis option
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://images.livemint.com/img/2021/07/16/1600x900/Reuters_1626422055339_1626422062780.jpg', use_column_width=True)

# Load the PNG image
# background_image = Image.open("oly.png") 


# st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')
selected_option = st.sidebar.radio('Select an Option', ('Olympic_Medals','olympics_medal_predictor', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis'))
st.image('OIP.jpeg', width=150)


# Perform analysis based on user selection
if selected_option == 'Olympic_Medals':
    st.title('Olympic Medals Analysis')
    years, country = olympic_medal_analysis.country_year_lst(df)
    selected_year = st.selectbox("Select Year", years)
    selected_country =st.selectbox("Select Country", country)
    medals = olympic_medal_analysis.olympic_medals(df, selected_year, selected_country)

   
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    
  
    # Apply custom styling to the table headers
    styled_data = medals.style.set_table_styles([{'selector': 'th', 'props': [('color','#808080'), ('font-weight', 'bold')]}])

    # styled_data = medals.style.set_table_styles([{'selector': 'th', 'props': [('color', '#050505'), ('font-weight', 'bold')]}])
    st.table(styled_data)
    # Display the table with bold column headers
    # st.table(medals.style.format({col: lambda x: f"{x}" for col in medals.columns}))




if selected_option == 'Overall Analysis':
    
    # Calculate statistics
    sports = df['Sport'].nunique()
    events = df['Event'].nunique()
    editions = df['Year'].nunique() - 1  # Subtract 1 to exclude the current year
    cities = df['City'].nunique()
    athletes = df['Name'].nunique()
    nations = df['region'].nunique()
    # Display image with specified width 
    
    # Title
    st.title("Olympic Insights")
    

    # Sidebar navigation
    selected_section = st.selectbox("Select Section", ["Overview", "Participating Nations", "Events Over Time", "Athletes Over Time", "Events Over Time (Every Sport)", "Most Successful Athletes"])

    if selected_section == "Overview":

        # Overview section
        st.title("Olympic Data Analysis Overview")
        st.write("""
        Welcome to the Olympic Data Analysis dashboard! Here, you can explore various insights derived from historical Olympic data.
        
        - **Participating Nations:** Discover the trends in the number of nations participating in the Olympics over the years.
        - **Events Over Time:** Visualize the evolution of Olympic events across different editions.
        - **Athletes Over Time:** Track the growth in the number of athletes competing in the Olympics over the years.
        - **Events Over Time (Every Sport):** Explore the number of events held for each sport over different editions.
        - **Most Successful Athletes:** Find out who the most successful athletes are based on their medal counts.
        
        Use the dropdown menu above to navigate through these sections and uncover fascinating insights about the Olympics!
        """)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Editions")
            st.metric(label="Total", value=editions, delta=None)

        with col2:
            st.subheader("Hosts")
            st.metric(label="Total", value=cities, delta=None)

        with col3:
            st.subheader("Sports")
            st.metric(label="Total", value=sports, delta=None)

        col4, col5, col6 = st.columns(3)

        with col4:
            st.subheader("Events")
            st.metric(label="Total", value=events, delta=None)

        with col5:
            st.subheader("Nations")
            st.metric(label="Total", value=nations, delta=None)

        with col6:
            st.subheader("Athletes")
            st.metric(label="Total", value=athletes, delta=None)

    elif selected_section == "Participating Nations":
        # Participating Nations over the years
        nations_over_time = olympic_medal_analysis.data_over_time(df, 'region')
        fig = px.line(nations_over_time, x="Edition", y="region")
        st.title("Participating Nations over the years")
        st.plotly_chart(fig)

    elif selected_section == "Events Over Time":
        # Events over the years
        events_over_time = olympic_medal_analysis.data_over_time(df, 'Event')
        fig = px.line(events_over_time, x="Edition", y="Event")
        st.title("Events over the years")
        st.plotly_chart(fig)

    elif selected_section == "Athletes Over Time":
        # Athletes over the years
        athlete_over_time = olympic_medal_analysis.data_over_time(df, 'Name')
        fig = px.line(athlete_over_time, x="Edition", y="Name")
        st.title("Athletes over the years")
        st.plotly_chart(fig)

    elif selected_section == "Events Over Time (Every Sport)":
        # No. of Events over time (Every Sport)
        st.title("Events Over Time (Every Sport)")
        x = df.drop_duplicates(['Year', 'Sport', 'Event'])
        heatmap_data = x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int')
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(heatmap_data, annot=True, ax=ax)
        st.pyplot(fig)

    elif selected_section == "Most Successful Athletes":
        # Most successful Athletes
        st.title("Most Successful Athletes")
        sport_list = df['Sport'].unique().tolist()
        sport_list.sort()
        sport_list.insert(0, 'Overall')

        selected_sport = st.selectbox('Select a Sport', sport_list)
        SX = olympic_medal_analysis.most_successful(df, selected_sport)
        st.table(SX)

if selected_option == "Country-wise Analysis":

    st.title("Country-wise Analysis")

    country_list = df["region"].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.selectbox("Select a Country",country_list)

    country_df = olympic_medal_analysis.yearwise_medal_tally(df,selected_country)

    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    
    pt =olympic_medal_analysis.country_event_heatmap(df,selected_country)
    if pt is not None:
        st.title(selected_country + " excels in the following sports")
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt,annot=True, cmap="cubehelix")
        st.pyplot(fig)
    
    st.title("Top 10 athletes of " + selected_country)
    top10_df = olympic_medal_analysis.most_successful_countrywise(df,selected_country)
    
    st.table(top10_df)

if selected_option == 'Athlete wise Analysis':

    athlete_df = df.drop_duplicates(subset=["Name", "region"])

    x1 = athlete_df["Age"].dropna()
    x2 = athlete_df[athlete_df["Medal"] == "Gold"]["Age"].dropna()
    x3 = athlete_df[athlete_df["Medal"] == "Silver"]["Age"].dropna()
    x4 = athlete_df[athlete_df["Medal"] == "Bronze"]["Age"].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ["Overall Age", "Gold Medalist", "Silver Medalist", "Bronze Medalist"],
                             show_hist=False, show_rug=False)
    #fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    #fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    st.title("Height-Vs-Weight")
    sport_list = df["Sport"].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
    selected_sport = st.selectbox("Select a Sport",sport_list)
    temp_df = olympic_medal_analysis.weight_v_height(df,selected_sport)
    fig,ax = plt.subplots(figsize=(10,10))
    ax = sns.scatterplot(x=temp_df["Weight"],y=temp_df["Height"],hue=temp_df["Medal"],style=temp_df["Sex"],s=100)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = olympic_medal_analysis.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    #fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

if selected_option == 'olympics_medal_predictor':
    olympics_medal_predictor()    
    

    # # Define available options
    # sport_options = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon', 'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping', 'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']
    # country_options = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA', 'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']

    # # Medal Predictor
    # st.title("Olympics Medal Predictor")

    # # User input widgets
    # with st.form("my_form"):
    #     st.header("Enter Athlete Details")
    #     Sex = st.selectbox("Sex", ["M", "F"])
    #     Age = st.slider("Age", min_value=10, max_value=97, step=1)
    #     Height = st.slider("Height (cm)", min_value=127, max_value=226, step=1)
    #     Weight = st.slider("Weight (kg)", min_value=25, max_value=214, step=1)
    #     region = st.selectbox("Country", country_options)
    #     Sport = st.selectbox("Sport", sport_options)
    #     input_model = st.selectbox("Prediction Model", ["Random Forest Classifier", "Logistic Regression", "Neural Network"])
    #     submitted = st.form_submit_button("Predict")

    # if submitted:
    #     # Prepare input data
    #     inputs = pd.DataFrame([[Sex, region, Sport, Height, Weight, Age]], columns=["Sex", "region", "Sport", "Height", "Weight", "Age"])
    #     print(inputs)
    #     # Transform input data
    #     inputs_transformed = data_pipeline.transform(inputs)

    #     #Select prediction model
    #     if input_model == "Random Forest Classifier":
    #         model = random_forest
    #     if input_model == "Logistic Regression":
    #         model = logistic_regression
    #     if input_model == "Neural Network":
    #         model = ann_model

    #     print(inputs_transformed)
    #     # Make prediction
    #     # prediction = random_forest.predict(inputs_transformed)
       

    #     # Display prediction result
    #     st.subheader("Prediction Result")
    #     # if prediction[0] == 0:
    #     #     st.error(f"The probability of winning a medal is low ({probability[0]:.2f})")
    #     # else:
    #     #     st.success(f"The probability of winning a medal is high ({probability[0]:.2f})")


       

    #     # Make prediction
    #     prediction = model.predict(inputs_transformed)
    #     probability = model.predict_proba(inputs_transformed)[:, 1]  # Probability of winning (class 1)
    #     win_percentage = probability[0] * 100

    #     if win_percentage < 40:
    #         win_label = "Low"
    #     elif win_percentage >= 40 and win_percentage < 70:
    #         win_label = "Moderate"
    #     else:
    #         win_label = "High"

    #     if prediction[0] == 0 :                    
    #         st.warning("Medal winning probability is {} ({}%)".format(win_label, round(win_percentage, 2)), icon="⚠️")
    #     else :
    #         st.success("Medal winning probability is {} ({}%)".format(win_label, round(win_percentage, 2)), icon="✅")
    #     # if prediction[0] == 0 :                    
    #     #     ans = "Low"
    #     #     st.warning("Medal winning probability is {}".format(ans),icon="⚠️")
    #     # else :
    #     #     ans = "High"
    #     #     st.success("Medal winning probability is {}".format(ans),icon="✅")