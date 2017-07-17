# Demos-for-sales-marketting

This includes two imagenet classifiers, VGG16 and inception. 

## Running the demos

### Download the needed files

The release will have the follwoing files, you will need to download the model you wish to use and place in your /tmp folder. 

trained_inception_model.zip
vgg16.zip

Inception is 84MB, and vgg16 is 490M. 

You will also need to download the jar 

training-demos-1.0-SNAPSHOT-bin.jar 

### Running the application

To run the web application simply run this command


```
java -cp training-demos-1.0-SNAPSHOT-bin.jar ai/skymind/training/InceptionWebApp
```

This will launch a web application listening on the following urls. 

http://localhost:4567/hello 

This just returns the string "Hello World" and is useful for debugging. 

http://localhost:4567/predict

This will return a form asking the user to select an image and providing an "Upload Picture" submit button. When an image is selected and "Upload Picture" is clicked the user will be taken to a results page showing the imagenet predictions. 

http://localhost:4567/getPredictions

Here is an example of the json output. 

{"data":[{"label":"Indian_elephant","prediction":90.71257},{"label":"tusker","prediction":4.065172},{"label":"African_elephant","prediction":1.4640747},{"label":"water_buffalo","prediction":1.0157301},{"label":"triceratops","prediction":0.7771906}], "performance":{ "feedforward":961,"total":1291}, "network":{ "parameters":23660968,"layers":203}}

