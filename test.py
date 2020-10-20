from preprocess import generate_bow
from kmeans import KMeans

data = generate_bow('./data.txt')
# print(data)
km = KMeans(data)
# print(km.closest_ct())
km.fit()