# MIDL-Assignment
Create a classifier model in Jupyter Notebook using Pytorch to detect color in an image

Data_Files (Link: https://drive.google.com/file/d/1rpooEkZ198pWw34eK9SvT_aMhPHK89_V/view?usp=sharing):

    -Images
    -Images.zip
    -test_new
      -_annotations.csv
    -styles.csv
  
Python version: 3.10.12

To run:

1. Clone repo or download notebook
2. Download and unzip Data_Files from link provided
3. Place Data_Files in same directory as Image_Color_Detection.ipynb
4. Open Image_Color_Detection.ipynb and run all

The code will first load in style.csv and create a data frame (image id, image color class) for sorting the images into their respective classes.

We will create a new folder called Data in the following format:

        Data
            Black
                Img1
                Img2
            Blue
                Img3
                Img4
            Brown
                Img5
                Img6

Once loaded into this file format we will transform (resize, flip, convert to tensor) and load the images into a data frame. 
Our network architecture is as follows (references VGG architecture):

    ColorDetectionModel(
  
      (conv_block_1): Sequential(
        (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(2, 2), stride=(1, 1), padding=(2, 2))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      
      (conv_block_2): Sequential(
        (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      
      (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=3610, out_features=46, bias=True)
      )
    )

The model gets trained with the train/test data from the Data folder. We achieve a training accuracy of approx. 64% and a test/validation accuracy of approx. 59%

To further test the model we load images from a new dataset (test_new) and see how our model predicts the color.

In 'Testing on New Data' section will create 

        New Images
            Black
                Img1
                Img2
            Blue
                Img3
                Img4
            Brown
                Img5
                Img6
                
Common Issues:

1. On older versions of Python use:
  df = pd.read_csv(..., error_bad_lines=False) 
  On new versions use the following:
  df = pd.read_csv(..., on_bad_lines='skip')
2. Watch out for where files and folders are being stored. We might need to adjust file path in certain areas if the code can't find the file
3. When you move the images (want to reduce memory usage so not copying) to their new classified directories (Data, New Images) they will no longer be in their original files (images, test_new). So if you re-run the code to move the images non will be found. To fix this we provided zip files for the files images and test_new. Simply unzip the files to replace the empty ones 


