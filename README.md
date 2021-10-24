# UNSW COMP9517 21T3 Project

### Instructions for collaborating on Google Colab and Git Hub:
Author: Hengrui Wang

#### General steps for executing code on Google Colab:
1. Use the unzipped dataset("Sequences" folder) on Git Hub
2. Open notebook on Google Colab, click on "Runtime" at top tool bar, select "Change runtime type", select "GPU" (If you want to use CPU, ignore this step but make sure to change "--device" option to "cpu" when training and testing)
3. click on Files icon on left side, click mount google drive
4. Modify the dataset path after "!unzip " to your dataset path in Google Drive
5. Execute cells

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
4. and modify the code
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
7. Execute cells! (e.g. "!python train.py")

#### Notes on useful git commands:
1. Create branch and checkout to that branch:
    ```bash
    # create new branch on git
    git checkout -b branch_name
    # move to branch
    git checkout branch_name
    ```
2. Merge **branch2** into **branch1**[ref_link](https://stackoverflow.com/questions/37709298/how-to-get-changes-from-another-branch):
    ```bash
    # move to branch2
    git checkout branch2
    # pull all new changes
    git pull 
    # move to branch1
    git checkout branch1
    # pull all new changes
    git pull 
    # merge two changes
    git merge branch2
    # push the merged branch2 to result
    git push
    ```
