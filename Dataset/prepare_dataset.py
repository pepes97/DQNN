import os

os.chdir("foa_dev")
file_list = os.listdir()
file_list.sort()

for ovo in [1, 2]:
  ov = f"ov{ovo}"
  for splito in [1,2,3,4]:
    
    split = f"split{splito}"
    try:
      os.mkdir(f"wav_ov{ovo}_split{splito}_30db")
      os.mkdir(f"desc_ov{ovo}_split{splito}")
    except FileExistsError:
      pass

    for f in file_list:
      if ov in f and split in f:
        print(f)
        os.rename(f, f"wav_ov{ovo}_split{splito}_30db/{f}")
        print(f[:-4])
        os.rename(f"../metadata_dev/{f[:-4]}.csv", f"desc_ov{ovo}_split{splito}/{f[:-4]}.csv")



import random

folders = os.listdir()
for folder in folders:
  print("\n"+folder)
  files = os.listdir(f"{folder}")
  print(f"{len(files)} in folder {folder}")
  files.sort()
  random.seed(42)
  test = random.sample(files, k = (len(files)//5))
  for elem in test:
    files.remove(elem)
  train = files
  print(f"{len(train)} element in train")
  print(f"{len(test)} element in test")
  
  for x in test:
    for y in train:
      assert x!=y

  train.sort()
  for i, elem in enumerate(train):
    if "wav" in folder:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"train_{i}_desc_30_100.wav"))
    else:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"train_{i}_desc_30_100.csv"))
  
  test.sort()
  for i, elem in enumerate(test):
    if "wav" in folder:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"test_{i}_desc_30_100.wav"))
    else:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"test_{i}_desc_30_100.csv"))

os.chdir("..")
os.rename("foa_dev", "TAU Dataset")
os.rmdir("metadata_dev")