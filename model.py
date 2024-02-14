
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer



# Load data
df = pd.read_excel(r"C:\Users\aravi\OneDrive\Desktop\Project Files\Deployment\World_development_mesurement.xlsx")
df1=df.copy()
# Creating a function to handle string characters and convert the non numeric into float
def Stringfunction(x):
    if isinstance(x, str):
        x = x.replace('$','')
        x = x.replace(',', '')
        x = x.replace('%', '')
        x = float(x)
    elif isinstance(x, float):
        pass  # no need to do anything if it's already a float
    else:
        try:
            x = x.replace('$','')
            x = x.replace(',', '')
            x = x.replace('%', '')
            x = float(x)
        except:
            pass
    return x
df=df.drop('Country', axis=1)
df = df.applymap(Stringfunction) # Applymap aplies function to each element of the dataframe
df['Country']=df1['Country']
 # Dropping unnecessary columns
df = df.drop(['Number_of_Records', 'Ease_of_Business'], axis=1)

# Handling missing values
imputer = KNNImputer(n_neighbors=3)
df_impute = df.drop('Country', axis=1)
imputed = imputer.fit_transform(df_impute)
df_imputed = pd.DataFrame(imputed, columns=df_impute.columns)

# Dropping features with high missing values, unnecessary features
df_imputed = df_imputed.drop(['Business_Tax_Rate', 'Hours_to_do_Tax', 'Days_to_Start_Business','Lending_Interest','Health_Exp/Capita'], axis=1)
df_imputed['Country']=df1['Country']
# Handling outliers using IQR
for col in df_imputed.columns:
    if col != 'Country':
        Q1 = df_imputed[col].quantile(0.25)
        Q3 = df_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_imputed[col] = np.where(df_imputed[col] < lower_bound, lower_bound, df_imputed[col])
        df_imputed[col] = np.where(df_imputed[col] > upper_bound, upper_bound, df_imputed[col])

# Scaling data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_imputed.drop('Country', axis=1))

# PCA for dimensionality reduction
pca = PCA(n_components=4) 
pca_values = pca.fit_transform(scaled_data)
pca_data = pd.DataFrame(pca_values, columns=['pc1', 'pc2', 'pc3', 'pc4'])
pca_data=np.array(pca_data)
# Hierarchical Clustering
kmeans_pca = KMeans(n_clusters=3,random_state=0)
kmeans_pca.fit(pca_data)


# Assigning labels to the data
labels = kmeans_pca.labels_
df['Cluster'] = labels
print(df['Cluster'])

model = {'scaler': scaler, 'pca': pca, 'kmean': kmeans_pca}
with open('trained_model_clustering.pkl', 'wb') as f:
    pickle.dump(model, f)

####
