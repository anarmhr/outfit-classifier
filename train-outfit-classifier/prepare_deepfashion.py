import os
import pandas as pd

dirs = os.listdir("./datasets/deep_fashion/img")

labels = []

for dir in dirs:
    labels.append(dir.split('_')[-1])

labels = [*set(labels)]
labels.sort()

# label encoding
df = pd.DataFrame(labels, columns=['label'])
df.to_csv('./datasets/deep_fashion/labels.csv', index_label='id')

dirs = os.listdir("./datasets/deep_fashion/img")

dirs.sort()

data = {}

print(os.listdir("./datasets/deep_fashion/img/" + dirs[0]))

data = []
for i in range(len(dirs)):
    label = labels.index(dirs[i].split('_')[-1])
    for img in os.listdir("./datasets/deep_fashion/img/" + dirs[i]):
        row = {'path': dirs[i] + '/' + img, 'label': label}
        data.append(row)


df = pd.DataFrame.from_dict(data)
df.to_csv('./datasets/deep_fashion/data.csv')
print(df)