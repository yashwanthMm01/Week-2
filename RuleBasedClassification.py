def classify_waste(waste_item):
    organic = ["banana peel", "apple core", "food waste", "vegetable"]
    recyclable = ["plastic bottle", "paper", "cardboard", "newspaper"]
    hazardous = ["battery", "broken glass", "paint", "chemicals"]

    waste_item = waste_item.lower()
    if waste_item in organic:
        return "Organic"
    elif waste_item in recyclable:
        return "Recyclable"
    elif waste_item in hazardous:
        return "Hazardous"
    else:
        return "Unknown"

# Test
print(classify_waste("Plastic Bottle"))  # Output: Recyclable
print(classify_waste("Old Battery"))     # Output: Hazardous
