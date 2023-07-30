import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import plotly.graph_objects as go
from sklearn.manifold import TSNE



df_merged = pd.read_parquet("df_merged.parquet")
import streamlit as st

# Define all your pages
def page1():
    st.title("Data")
    st.write("Welcome to Data Page")

def page2():
    st.title("t-SNE")
    st.write("Welcome to t-SNE Page")

    if st.button('Run t-SNE'):

        # Select the features
        features = df_merged[['overall','reviewText_sentiment','summary_sentiment','average_price','season']]

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
                color=df_merged['overall'],  # set color to season
                colorscale='Viridis',  # choose a colorscale
                colorbar=dict(title="Overall review"),  # colorbar title
                opacity=0.8
            )
        )])

        st.plotly_chart(fig)


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Page 1", "Page 2"]) # Add more pages here

    if selection == "Page 1":
        page1()
    elif selection == "Page 2":
        page2()
    # Add more pages here in elif statements

if __name__ == "__main__":
    main()



st.write('Hello world!')

