#import the required libraries
import pandas as 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit


# load the dataset
df = pd.read_csv('API_SM.POP.NETM_DS2_en_csv_v2_5358390.csv', skiprows=4)

#Applying the methods From Practical class

def scaler(df):
   
    # Use the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max

def backscale(arr, df_min, df_max):

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1960, 2010)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True) # replace NaN values with 0

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Deciding the number of clusters to use
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)

# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['2005'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('Net Migration Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('Net Migration in 2005')

# Add legend
plt.legend()

# Show the plot
plt.show()



# Display countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(cluster_countries)
    print()


#Checking the years in the list
print(df_years.columns)


def linear_model(x, a, b):
    return a*x + b

# Define the columns to use
columns_to_use = [str(year) for year in range(1960, 2009)]


# Select a country
country = 'United States'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values[0]
x_data = np.array(range(1960, 2009))
y_data = country_data

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)

def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper

# Predicting future values and corresponding confidence Ranges
x_future = np.array(range(1960, 2070))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, alpha=0.3, label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Net Migration')
plt.title(f'{country} Net Migration Fitting')
plt.legend()
plt.show()


#ploting a line graph to showcase top five regions with the highest net migration

# Filter the data to include only the top 5 countries with highest net migration
df_top5 = df.sort_values(by='2019', ascending=False).head(5)

# Melt the dataframe to convert years into a single column
df_melt = pd.melt(df_top5, id_vars='Country Name', value_vars=[str(i) for i in range(2015, 2022)], var_name='year', value_name='value')

# Convert the value column to numeric
df_melt['value'] = pd.to_numeric(df_melt['value'], errors='coerce')

# Filter the data to include only the years in the range 2015-2021
df_melt = df_melt[(df_melt['year'] >= '2015') & (df_melt['year'] <= '2021')]

# Plot the data as a line graph
plt.figure(figsize=(10, 6))
for country in df_top5['Country Name']:
    df_country = df_melt[df_melt['Country Name'] == country]
    plt.plot(df_country['year'], df_country['value'], label=country)
plt.title('Top 5 Countries with Highest Net Migration')
plt.xlabel('Year')
plt.ylabel('Net Migration')
plt.legend()
plt.show()
