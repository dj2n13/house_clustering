import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

excel_file_path = "data/boroughs.xlsx"
result_columns = ["id", "Borough", "avg_price", "crime_rate"]
selected_columns = ["avg_price", "crime_rate"]


# Function to load data
def load_data(file_path, usecols=None):
    try:
        data = pd.read_excel(file_path, usecols=usecols)
        return data
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        return None


# Function to normalize data
def preprocess_data(df_selected):
    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df_selected)
    return pd.DataFrame(df_scaled, columns=df_selected.columns), scaler


# Function to plot scatter
def plot_data(
    data, x, y=None, title="", xlabel="", ylabel="", kind="scatter", hue=None
):
    plt.figure()
    if kind == "scatter":
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
    elif kind == "line":
        plt.plot(data[x], data[y] if y else data, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Function to determine the optimal number of clusters
def calculate_optimal_clusters(data, range_clusters):
    wcss = []
    silhouette_scores = []
    for i in range_clusters:
        kmeans = KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        if i > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
    return wcss, silhouette_scores


# Function to label clusters
def label_clusters(row, median_price, median_crime_rate):
    avg_price = row["avg_price"]
    crime_rate = row["crime_rate"]
    if avg_price >= median_price and crime_rate < median_crime_rate:
        return "High Price, Low Crime"
    elif avg_price < median_price and crime_rate >= median_crime_rate:
        return "Low Price, High Crime"
    elif avg_price >= median_price and crime_rate >= median_crime_rate:
        return "High Price, High Crime"
    else:
        return "Low Price, Low Crime"


# Preprocess the data
def main():
    df_result = load_data(excel_file_path, usecols=result_columns)
    df_selected = load_data(excel_file_path, usecols=selected_columns)
    if df_result is None or df_selected is None:
        return
    df_scaled, scaler = preprocess_data(df_selected)

    # Determine the optimal number of clusters using the Elbow Method
    wcss, silhouette_scores = calculate_optimal_clusters(
        df_scaled[selected_columns], range(1, 32)
    )
    elbow_data = pd.DataFrame({"Number of Clusters": range(1, 32), "WCSS": wcss})

    plot_data(
        elbow_data,
        "Number of Clusters",
        "WCSS",
        title="The Elbow Method",
        xlabel="Number of Clusters",
        ylabel="WCSS",
        kind="line",
    )

    # Plotting Silhouette Scores
    silhouette_data = pd.DataFrame(
        {"Number of Clusters": range(2, 32), "Silhouette Score": silhouette_scores}
    )
    plot_data(
        silhouette_data,
        "Number of Clusters",
        "Silhouette Score",
        title="Silhouette Scores for Different Numbers of Clusters",
        xlabel="Number of Clusters",
        ylabel="Silhouette Score",
        kind="line",
    )
    # Perform KMeans clustering with the selected number of clusters
    num_clusters = 12
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=0,
    )
    df_scaled["cluster"] = kmeans.fit_predict(df_scaled[selected_columns])
    df_scaled[selected_columns] = scaler.inverse_transform(df_scaled[selected_columns])

    # Compute median values for price and crime
    global median_price, median_crime_rate
    median_price = df_scaled["avg_price"].median()
    median_crime_rate = df_scaled["crime_rate"].median()

    # Apply labels
    df_scaled["label"] = df_scaled.apply(
        lambda row: label_clusters(row, median_price, median_crime_rate), axis=1
    )

    # Visualize the clusters with labels
    plot_data(
        df_scaled,
        "avg_price",
        "crime_rate",
        title="Clustered Data with Labels",
        xlabel="Average Price",
        ylabel="Crime Rate",
        kind="scatter",
        hue="label",
    )

    # Export the clustered and labeled data
    df_result["label"] = df_scaled["label"]
    df_result["cluster"] = df_scaled["cluster"]
    output_file_path = "data/labeled_clustered_boroughs.csv"
    df_result.to_csv(output_file_path, index=False)


main()
