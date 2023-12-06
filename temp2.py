from roboflow import Roboflow

rf = Roboflow(api_key="nFfIEBI0ZbgVas8BHUBt")
project = rf.workspace().project("cable-detection-liolt")
model = project.version(1).model

# infer on a local image
# print(model.predict("your_image.jpg").json())

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE").json())

# save an image annotated with your predictions
model.predict("3_00092.jpg").save("prediction.jpg")