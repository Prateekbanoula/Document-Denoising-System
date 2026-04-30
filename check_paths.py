import os

train_path = "C:/Users/prate/Downloads/train (1)/train/train_shabby/"
clean_path = "C:/Users/prate/Downloads/train (1)/train/train_cleaned/"

train_files = set(os.listdir(train_path))
clean_files = set(os.listdir(clean_path))

common = train_files & clean_files

print("Train:", len(train_files))
print("Clean:", len(clean_files))
print("Matching:", len(common))