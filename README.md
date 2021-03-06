### Acknowledgement of open source code used
The code for machine learning pipeline is partially adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html and https://github.com/cosmic-cortex/pytorch-UNet

The code skeleton for dataset is adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and https://github.com/cosmic-cortex/pytorch-UNet

Source of UNet model: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

Source of VGG model: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py

The code for DBSCAN clustering is adapted from https://cs.stackexchange.com/questions/85929/efficient-point-grouping-algorithm/86040

Corresponding acknowledgements are also listed on the top of the file if open source code is used.

### Sub tasks for completing project
- [x] Task 1: Dataset Processing
- [x] Task 2: Cell Segmentation(ml)
- [x] Task 3: Cell Segmentation(alg)
- [x] Task 4: Instance Segmentation([watershed based?](https://www.youtube.com/watch?v=lOZDTDOlqfk))
- [x] Task 5: Cell Tracking(alg)
- [x] Task 6: Mitosis Detection(ml)
- [x] Task 7: Mitosis Detection(alg)

### List of utility functions required to complete above tasks(completed functions in utils.py)

- [x] 1. Generate pseudo masks(png, 255 for cells, 0 for background) using watershed algorithm and save to git hub for experiment on other functions
- [x] 2. Extract tracking information from TRA images. prob output: Image and dict(key:unique label, value:central coord on image and corresponding information about mitosis)
- [x] 3. Process tif images to png format
- [x] 4. Write function to remove cells on the image boundary
- [x] 5. Write function to read 2d np.array mask(255 for cell, 0 for background) as input and output a list of central points for all cells in the mask
- [x] 6. Write function to calculate the mean size of cells by giving a 2d np.array mask
- [x] 7. Write function to extract contours of cells from mask
- [x] 8. Write function to calculate the average displacement of all cells between two masks
- [x] 9. Write function to convert mask to object instances image(Similar to images in TRA folder, each cell have a unique label)
- [x] 10. Write function to evaluate the predicted mask and ground truth mask
- [x] 11. Write function to combine sequence of images into .gif for better visualization

### Visualization
Input Image                | Instance Segmentation
:-------------------------:|:-------------------------:
<img src="./readme_imgs/02_input.gif" width="440" height="280"> | <img src="./readme_imgs/02_pred.gif" width="440" height="280">

Cell Tracking              | Mitosis Detection
:-------------------------:|:-------------------------:
<img src="./readme_imgs/02_path.gif" width="440" height="280"> ??? <img src="./readme_imgs/02_boundary.gif" width="440" height="280">

Task1                      | Task2
:-------------------------:|:-------------------------:
<img src="./readme_imgs/task2.png" width="440" height="280"> | <img src="./readme_imgs/task3.jpg" width="440" height="280">

### Instructions for collaborating between Google Colab and Git Hub:
Author: Hengrui Wang

#### General steps for executing code on Google Colab:
1. Use the unzipped dataset("Sequences" folder) on Git Hub
2. Open notebook on Google Colab, click on "Runtime" at top tool bar, select "Change runtime type", select "GPU" 
(If you want to use CPU, ignore this step but make sure to change "--device" option to "cpu" when training and testing)

3. Run cells

#### General steps for modifying code on git and run modified code on Colab:
1. Clone git repo to your laptop:
    Use option A in this [link](https://stackoverflow.com/questions/651038/how-do-you-clone-a-git-repository-into-a-specific-folder)

    Having trouble finding the https git repo link? see following instructions:

    1. On the GitHub website, click on you repository of interest.

    2. Locate the button named "Code" and click on it. The GitHub URL will appear.

    3. Copy the GitHub URL

    4. Use the command described in option A: "git clone link folder-name"

2. Open the folder with your favorite IDE 
3. Pull the git repo to see if there are any new changes
    ```bash
    git pull
    ```
4. Modify code
5. In order to upload modified code to git:
    1. Save changes in your IDE
    2. Open terminal and move to your project folder
    3. Commit changes: (**Make sure to notify authors before changing their code**)
        ```bash
        git commit -m "commit message"
        ```
    4. Push changes to git hub
        ```bash
        git push
        ```
6. On Colab notebook, run the cell contains code "!git pull" to get updated code from git hub
7. Run cells (e.g. "!python train.py")

#### Notes on useful git commands:
1. Create branch and checkout to that branch:
    ```bash
    # create new branch on git
    git checkout -b branch_name
    # move to branch
    git checkout branch_name
    # commit and push to git
    git commit -m "commit_message"
    git push origin branch_name
    ```
2. Merge **branch2** into **branch1**, [ref_link](https://stackoverflow.com/questions/37709298/how-to-get-changes-from-another-branch):
    ```bash
    # move to branch2
    git checkout branch2
    # pull all new changes
    git pull 
    # move to branch1
    git checkout branch1
    # pull all new changes
    git pull 
    # merge two branches
    git merge branch2
    # push the merged branch2 to result
    git push
    ```
