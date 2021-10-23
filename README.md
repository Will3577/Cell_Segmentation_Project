# UNSW COMP9517 21T3 Project

### Instructions for collaborating on Google Colab and Git Hub:

#### General steps for executing the code on Google Colab:
1. Upload dataset.zip to Google drive(any questions, see this [link](https://support.google.com/drive/answer/2424368?hl=en&co=GENIE.Platform%3DDesktop))
2. Open notebook on Google Colab, click on "Runtime" at top tool bar, select "Change runtime type", select "GPU"
3. click on Files icon on left side, click mount google drive
4. Modify the dataset path after "!unzip " to your dataset path in Google Drive
5. Execute cells

#### General steps for modifying code on git and run modified code on Colab:
1. Clone git repo to your laptop:
    Use option A in this [link](https://stackoverflow.com/questions/651038/how-do-you-clone-a-git-repository-into-a-specific-folder)

    Having trouble finding the https repo link?

        1. On the GitHub website, click on you repository of interest.

        2. Locate the button named "Code" and click on it. The GitHub URL will appear.

        3. Copy the GitHub URL
        
        4. Use the command described in option A: "git clone link folder-name"

2. Open the folder with your favorite IDE and modify the code
3. In order to upload modified code to git:
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
4. On Colab notebook, run the cell contains code "!git pull" to get updated code from git hub
5. Execute cells (e.g. "!python train.py")

