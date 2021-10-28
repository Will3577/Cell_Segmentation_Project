# UNSW COMP9517 21T3 Project

### Sub tasks for completing project
- [x] Task 1: Dataset Processing
- [x] Task 2: Cell Segmentation(ml)
- [x] Task 3: Cell Segmentation(alg)
- [ ] Task 4: Instance Segmentation([watershed based?](https://www.youtube.com/watch?v=lOZDTDOlqfk))
- [ ] Task 5: Cell Tracking(alg)
- [ ] Task 6: Mitosis Detection(ml)
- [ ] Task 7: Mitosis Detection(alg)

### List of utility functions required to complete above tasks(completed functions in utils.py)

- [x] 1. Generate pseudo masks(png, 255 for cells, 0 for background) using watershed algorithm and save to git hub for experiment on other functions
- [x] 2. Process tif images to png format, intensity range: (0,255)
- [x] 3. Write function to remove cells on the image boundary
- [ ] 4. Write function to read 2d np.array mask(255 for cell, 0 for background) as input and output a list of central points for all cells in the mask
- [ ] 5. Write function to calculate the mean size of cells by giving a 2d np.array mask
- [x] 6. Write function to extract contours of cells from mask
- [ ] 7. Extract tracking information from TRA images. prob output: Image and dict(key:unique label, value:central coord on image and corresponding information about mitosis)
- [ ] 8. Write function to calculate the average displacement of all cells between two masks
- [ ] 9. Write function to convert mask to object instances image(Similar to images in TRA folder, each cell have a unique label)
- [ ] 8. Write function to evaluate the predicted mask and ground truth mask
- [ ] 9. Write function to combine sequence of images into .gif for better visualization



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
