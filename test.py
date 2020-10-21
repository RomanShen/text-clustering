from preprocess import generate_bow
from kmeans import KMeans
from hieclustering import HierarchicalClustering

data = generate_bow('./data.txt')

km = KMeans(data)
km.fit()

# hc = HierarchicalClustering(data)
# hc.fit()
