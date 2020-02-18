import os

os.chdir("foa_dev")
file_list = os.listdir()
file_list.sort()

for ovo in [1, 2]:
  ov = f"ov{ovo}"
    #for splito in [1,2,3,4]:
      
  splito = 0

  try:
    os.mkdir(f"wav_ov{ovo}_split{splito}_30db")
    os.mkdir(f"desc_ov{ovo}_split{splito}")
  except FileExistsError:
    pass

  for f in file_list:
    if ov in f:
      print(f)
      os.rename(f, f"wav_ov{ovo}_split{splito}_30db/{f}")
      print(f[:-4])
      os.rename(f"../metadata_dev/{f[:-4]}.csv", f"desc_ov{ovo}_split{splito}/{f[:-4]}.csv")

import random

folders = os.listdir()
for folder in folders:
  print("\n"+folder)
  files = os.listdir(f"{folder}")
  # print(f"{len(files)} in folder {folder}")
  # files.sort()
  # random.seed(42)
  # test = random.sample(files, k = (len(files)//5))
  # for elem in test:
  #   files.remove(elem)
  # train = files
  # print(f"{len(train)} element in train")
  # print(f"{len(test)} element in test")
  
  # for x in test:
  #   for y in train:
  #     assert x!=y
 
  # train.sort()

  files.sort()
  for i, elem in enumerate(files):
    if "wav" in folder:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"train_{i}_desc_30_100.wav"))
    else:
      os.rename(os.path.join(folder,elem), os.path.join(folder,f"train_{i}_desc_30_100.csv"))

os.chdir("..")



os.chdir("proj/asignal/DCASE2019/dataset/foa_eval")

try:
    os.mkdir("ov1")
    os.mkdir("ov2")
except FileExistsError:
    pass

for f in os.listdir("."):
    if f != "ov1" and f!= "ov2":
        print(f)
        pre = "split0_"
        post = ".wav"

        n = f[len(pre):-len(post)]
        n = int(n)
        if ((n-1)//10)%2==0:
            os.rename(f, f"ov1/{f}")
        else:
            os.rename(f, f"ov2/{f}")

os.chdir("../../../../..")
os.chdir("metadata_eval")
try:
    os.mkdir("ov1")
    os.mkdir("ov2")
except FileExistsError:
    pass

for f in os.listdir("."):
    if f != "ov1" and f!= "ov2":
        print(f)
        pre = "split0_"
        post = ".csv"

        n = f[len(pre):-len(post)]
        print(n)
        n = int(n)
        if ((n-1)//10)%2==0:
            os.rename(f, f"ov1/{f}")
        else:
            os.rename(f, f"ov2/{f}")
os.chdir("..")
os.chdir("proj/asignal/DCASE2019/dataset/foa_eval")
folders = os.listdir(".")

for folder in folders:
  lista = os.listdir(folder)
  lista.sort()
  for i, f in enumerate(lista):
    os.rename(os.path.join(folder,f), os.path.join(folder,f"test_{i}_desc_30_100.wav"))
    if folder=="ov1":
      fol = os.listdir("../../../../../foa_dev/")
      a = ""
      for x in fol:
        if "wav_ov1" in x:
          a = x
          break
      os.rename(os.path.join(folder,f"test_{i}_desc_30_100.wav"), "../../../../../foa_dev/"+a+"/"+f"test_{i}_desc_30_100.wav")
    else:
      fol = os.listdir("../../../../../foa_dev/")
      a = ""
      for x in fol:
        if "wav_ov2" in x:
          a = x
          break
      os.rename(os.path.join(folder,f"test_{i}_desc_30_100.wav"), "../../../../../foa_dev/"+a+"/"+f"test_{i}_desc_30_100.wav")

os.chdir("../../../../..")
os.chdir("metadata_eval")
folders = os.listdir(".")
for folder in folders:
  lista = os.listdir(folder)
  lista.sort()
  for i, f in enumerate(lista):
    os.rename(os.path.join(folder,f), os.path.join(folder,f"test_{i}_desc_30_100.csv"))
    if folder == "ov1":
      fol = os.listdir("../foa_dev/")
      a = ""
      for x in fol:
        if "desc_ov1" in x:
          a = x
          break
      os.rename(os.path.join(folder,f"test_{i}_desc_30_100.csv"), "../foa_dev/"+a+"/"+f"test_{i}_desc_30_100.csv")

    else:
      fol = os.listdir("../foa_dev/")
      a = ""
      for x in fol:
        if "desc_ov2" in x:
          a = x
          break
      os.rename(os.path.join(folder,f"test_{i}_desc_30_100.csv"), "../foa_dev/"+a+"/"+f"test_{i}_desc_30_100.csv")

os.chdir("..")
os.rename("foa_dev", "TAU Dataset")
os.rmdir("metadata_dev")
import shutil
shutil.rmtree('metadata_eval')
shutil.rmtree('proj')
