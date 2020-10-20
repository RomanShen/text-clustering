from preprocess import generate_bow
from kmeans import KMeans

data = generate_bow('./data.txt')
# print(data)
km = KMeans()
print(km.closest_ct(data))
# print(km.move_ct(data))