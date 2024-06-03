model = pickle.load(open('model.pkl','rb'))

def get_input():
    
    floor_size = int(input("Enter floor size: "))
    number_of_bathrooms = input("Enter number of bathrooms: ")
    number_of_bedrooms = int(input("Enter number of bedrooms: "))
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    year_built = input("Enter year built: ")
    garage = int(input("Enter number of garages: "))
    pool = input("Enter type of pool: ")
    
    d = {
    "floor_size": [floor_size],
    "number_of_bathrooms": [number_of_bathrooms],
    "number_of_bedrooms": [number_of_bedrooms],
    "latitude": [latitude],
    "longitude": [longitude],
    "year_built": [year_built],
    "garage": [garage],
    "pool": [pool],
    }
    
    df = pd.DataFrame(d)
    
    return df

def predict(housing):
    
    housing['pool'] = housing.apply(process_pool, axis = 1)
    housing['number_of_bathrooms'] = housing.apply(process_bathrooms, axis = 1)
    housing['year_built'] = housing.apply(process_year_built, axis = 1)
    housing.rename(columns = {"year_built": "age"}) 
    
    process_distance(housing)
    housing = pd.get_dummies(housing)
    
    return (model.predict(housing))

housing = get_input()
print(predict(housing))
