import requests
from bs4 import BeautifulSoup
import json
import pprint

urls = [str(f"https://www.bexrealty.com/Massachusetts/Boston/?page_number={i}") for i in range(1,41)]
dicts = []
year_urls = []
k = 0

for url in urls:
    
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    houses_info = soup.find_all('script', type='application/ld+json')
    
    
    for i in range (0, len(houses_info)):
        k+=1
        dicts.append(json.loads(str(houses_info[i].text)))

for house in dicts:
    try:
        year_urls.append(house[0]['mainEntity'][1]['offers']['url'])
    except:
        year_urls.append(None)

  new_dicts = []
k = 0

for url in year_urls:
    if url:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
    
        houses_info = soup.find_all('script', type='application/ld+json')
    
        for i in range (0, len(houses_info)):
            k+=1
            print(k)
            new_dicts.append(json.loads(str(houses_info[i].text)))
            
pprint.pprint(new_dicts[0])

names = []
postal_code = []
residence_type = []
floor_size = []
number_of_bathrooms_total = []
number_of_bedrooms = []
street_address = []
latitude = []
longitude = []
year_built = []
prices = []
garage = []
pool = []
house_type = []

for house in new_dicts:
    
    try:
        house_type.append(house[0]['amenityFeature'][0]['value'])
    except:
        house_type.append(None)
        
    try:
        garage.append(house[0]['amenityFeature'][1]['value'])
    except:
        garage.append(None)
        
    try:
        pool.append(house[0]['amenityFeature'][2]['value'])
    except:
        pool.append(None)

    try:
        residence_type.append(house[0]['@type'][1])
    except:
        residence_type.append(None)
    
    try:
        names.append(house[0]['name'])
    except:
        names.append(None)
    
    try:
        postal_code.append(house[0]['address']['postalCode'])
    except:
        postal_code.append(None)
    
    try:
        floor_size.append(house[0]['accommodationFloorPlan']['floorSize']['value'])
    except:
        floor_size.append(None)
    
    try:
        number_of_bathrooms_total.append(house[0]['numberOfBathroomsTotal'])
    except:
        number_of_bathrooms_total.append(None)
    
    try:
        street_address.append(house[0]['address']['streetAddress'])
    except:
        street_address.append(None)
    
    try:
        latitude.append(house[0]['geo']['latitude'])
    except:
        latitude.append(None)
    
    try:
        number_of_bedrooms.append(house[0]['numberOfBedrooms'])
    except:
        number_of_bedrooms.append(None)
    
    try:
        longitude.append(house[0]['geo']['longitude'])
    except:
        longitude.append(None)

    try:
        year_built.append(house[0]['yearBuilt'])
    except:
        year_built.append(None)

    try:
        prices.append(house[0]['offers']['price'])
    except:
        prices.append(None)

  import csv

with open('data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(names, prices, postal_code, residence_type, floor_size, number_of_bathrooms_total, number_of_bedrooms, street_address, latitude, longitude, year_built, garage, pool, house_type))
