from preprocess import generate_bow
from k_means import KMeans
from hieclustering_mean import HierarchicalMeanClustering

data = generate_bow('./data.txt')

# km = KMeans(data)
# km.fit()

hc = HierarchicalMeanClustering(data)
hc.fit()
