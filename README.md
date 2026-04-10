# Crime Hotspot Prediction

## Overview
This project implements an Artificial Intelligence (AI) model to predict and identify crime hotspots using spatial data. The model uses the **K-Means Clustering** algorithm to group crime locations and find the center of high-risk areas.

## Implementation Details
1. **Data Generation:** Synthetic spatial data (Latitude/Longitude) representing crime incidents was generated.
2. **AI Model:** The K-Means algorithm was applied to cluster the data into 5 main hotspots.
3. **Visualization:** The results are plotted using `matplotlib` and `seaborn`, with red 'X' marks indicating the predicted hotspot centers.

## How to Run the Code
1. Install the required libraries:
   `pip install numpy pandas scikit-learn matplotlib seaborn`
2. Run the Python script:
   `python main.py`
