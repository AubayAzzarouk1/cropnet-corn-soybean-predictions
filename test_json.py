import json 

with open("corn_train_soybean.json") as f:
    configure = json.load(f)

print("Loaded Crop Types: ",configure["crop_type"])

print("FIPS:",configure["crop_type"])
print("Years: ", configure["years"])

'''
JSON test on py: verified.
Loaded Crop Types:  ['Corn', 'Soybean']
FIPS: ['Corn', 'Soybean']
Years:  ['2022']
'''