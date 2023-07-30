import streamlit as st
import pandas as pd
import os
import gzip
import json
import string
import re
import numpy as np
import nltk
import openpyxl
import pyarrow
import fastparquet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import plotly.graph_objects as go
from sklearn.manifold import TSNE



df_merged = pd.read_parquet("df_merged.parquet")


# Add a button for running t-SNE
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
            color=df_merged['season'],  # set color to season
            colorscale='Viridis',  # choose a colorscale
            colorbar=dict(title="Seasons"),  # colorbar title
            opacity=0.8
        )
    )])

    # Display the figure in Streamlit
    st.plotly_chart(fig)


st.write('Hello world!')

