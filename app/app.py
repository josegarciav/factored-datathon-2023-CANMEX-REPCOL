import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pygwalker as pyg
import plotly.express as px  # pip install plotly-express
import math

st.set_page_config(page_title="Amazon reviews dashboard",
                   page_icon=":bar_chart:",
                   layout="wide",
                   initial_sidebar_state="auto",
                   )

# Import the dataframe merged and cache it
@st.cache_data
def get_data():    
    df = pd.read_parquet("df_merged.parquet")
    df["category"] = df["category"].apply(tuple)
    df["feature"] = df["feature"].apply(tuple)
    df["hour"] = pd.to_datetime(df["unixReviewTime"], unit="s").dt.hour
    return df

df = get_data()

def get_unique_list_values(df, column_name):
    """
    Method to obtain the unique values of list columns.
    """
    flattened_values = pd.Series([item for sublist in df[column_name] for item in sublist])
    unique_values = flattened_values.unique()
    return unique_values


# Define all your pages
def page1():
    # ---- SIDEBAR ----
    st.sidebar.header("Please Filter Here:")

    category_unique = get_unique_list_values(df, 'category')
    feature_unique = get_unique_list_values(df, 'feature')

    category = st.sidebar.multiselect(
        "Select the category:",
        options=category_unique,
        default=category_unique[:5]
    )

    feature = st.sidebar.multiselect(
        "Select the feature Type:",
        options=feature_unique,
        default=feature_unique[:5]
    )

    season = st.sidebar.multiselect(
        "Select the season:",
        options=df['season'].unique(),
        default=df['season'].unique(),
    )

    df_selection = df[
        df['category'].apply(lambda x: any([k in x for k in category])) &
        df['feature'].apply(lambda x: any([k in x for k in feature])) &
        df['season'].isin(season)
    ]


    # ---- MAINPAGE ----
    st.title(":bar_chart: Insights Dashboard")
    st.markdown("##")

    # TOP KPI's
    total_reviews = int(df_selection["reviewText"].count())
    average_rating = round(df_selection["overall"].mean(), 1)

    if math.isnan(average_rating):
        star_rating = ""
    else:
        star_rating = ":star:" * int(round(average_rating, 0))

    average_sale_by_transaction = round(df_selection["average_price"].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Reviews:")
        st.subheader(f"{total_reviews:,}")
    with middle_column:
        st.subheader("Average Rating:")
        st.subheader(f"{average_rating} {star_rating}")
    with right_column:
        st.subheader("Average price per item reviewed:")
        st.subheader(f"US $ {average_sale_by_transaction}")

    st.title("Filtered data")
    st.dataframe(df_selection)
    st.markdown("""---""") 


def page2():
    st.title("t-SNE")
    st.write("Welcome to the t-SNE dimensionality reduction Page")

    if st.button('Run t-SNE'):

        # Select the features
        features = df[['overall','reviewLength','reviewCount',
                       'reviewText_sentiment','summary_sentiment','average_price','season_num']]

        # Apply the t-SNE transformation
        tsne = TSNE(n_components=3, random_state=0)
        tsne_results = tsne.fit_transform(features)

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            z=tsne_results[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=df['overall'],  # set color to season
                colorscale='Viridis',  # choose a colorscale
                colorbar=dict(title="Overall review"),  # colorbar title
                opacity=0.8
            )
        )])

        st.plotly_chart(fig)
        st.text("We can see that overall the reviews group together by some latent feature.")


def main():
    st.sidebar.title("App sidebar")
    selection = st.sidebar.radio("Go to", 
                                 ["Page 1", "Page 2"]) # Add more pages here

    if selection == "Page 1":
        page1()
    elif selection == "Page 2":
        page2()
    # Add more pages here in elif statements

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()