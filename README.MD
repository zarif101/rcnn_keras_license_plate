# Basic Object Detect W/ Keras RCNN From Scratch

Please check out the following Medium article for more info!
ARTICLE STILL BEING WRITTEN

# The Basics
As I learned more and more about object detection, I decided to implement a basic version of a basic object detection algorithm - the RCNN. This implementation aims to detect license plates of vehicles, in image. Once again, please check out the Medium article for more info. I'm assuming you already know what you're getting into... :)

# Default Usage
IMPORTANT: SEE FIRST LINE OF 'Notes' SECTION!

Steps:
1. Enter the main directory via the terminal
2. Generate data using 'generate_data.py' - this will create the dataset, and place the images inside the 'use_data' folder. The folder will be empty before you run this.
```shell
python3 generate_data.py
```
3. Train the model that will classify image sections as 'plate' or 'not plate', using 'train_base_model.py'
```shell
python3 train_base_model.py
```
4. Run 'rcnn.py' to perform license plate detection on an image. By default, it will perform detection on 'many_plates.jpg' (provided in repo). To change the image, edit the following line
```python
main('many_plates.jpg','base_model.h5')
```
Run the file to perform detection! A labelled image will pop up!
```shell
python3 rcnn.py
```
# Custom Usage
IMPORTANT: SEE FIRST LINE OF 'Notes' SECTION!
- To run RCNN using a different base model (for example if you have a model trained on dog vs no dog), edit the following line:
```python
main('many_plates.jpg','base_model.h5')
```
- To change the parameters/hyper-parameters of the model, see 'models.py' and 'train_base_model.py'
- See the following section for more info!

# Notes
- As noted in the the code itself, you must edit the imports (at the top) section of all the '.py' files, to import Keras and different Keras elements, from Tensorflow or however you normally do. They currently import Keras via PlaidML, a framework that I use since I have an AMD GPU and can't use Tensorflow.
- I'm currently writing a Medium article to cover this project and the topic, in more detail. The link will be here once its up.
- I will soon  have a detailed demonstration video, in addition to the Medium article
- I'm quite busy, but I'll add more features/thorough documentation whenever I can.
- There is currently no command line args parser - it will be implemented soon, and you'll be able to change different parameters without directly editing the '.py' files.
- I'm working on a website where you'll be able to upload a photo, and see the results of the RCNN.
- Currently, you can only input one photo at a time, and it will simply show the detections on the photo. Feel free to experiment on your own, but I will be adding more flexibility in the future!
- There is currently no 'requirements.txt' file.
- The files will produce output(s) in the terminal
- The Jupyter Notebooks here simply document the process I went through to initially build the project.
- Credit to Adrian Rosebrock over at Pyimagesearch, for the non-maximum suppression and IOU (intersection over union) functions!
