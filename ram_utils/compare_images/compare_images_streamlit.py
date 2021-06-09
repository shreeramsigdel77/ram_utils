#streamlit run compare_images.py
import os
import cv2

import streamlit as st
import pandas as pd

st.write("""
# Compare Images
""")

col1,col2 = st.beta_columns(2)

file = col1.st.file_uploader("Pick a file",key="dir1")

file1 = st.file_uploader("Pick a file",key="dir2")

