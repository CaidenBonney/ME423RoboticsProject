# ME423RoboticsProject
Compilation of all code for the Cal Poly ME423 Robotics Project

# Requirements
## Python Version for venv: 
3.11.9

## Windows Installation Instructions:
### Go to Windows Powershell and paste the following:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop --version
scoop install git
scoop bucket add extras
scoop install uv
uv python install 3.11.9
uv python update-shell

### Create a virtual environment after opening the project in VS Code by pasting the following into the terminal:
uv run --python 3.11.9 python -m venv .venv

After creating the virtual environment, set the python interpreter to the virtual environment by going to VS Code's Search at the top:
">Python: Select Interpreter" > then select the virual environment you just created (says recommended next to it).

## Mac Installation Instructions:
### Go to Terminal and paste the following:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew update
brew install git
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9

### Create a virtual environment after opening the project in VS Code by pasting the following into the terminal:
python -m venv .venv

After creating the virtual environment, set the python interpreter to the virtual environment by going to VS Code's Search at the top:
">Python: Select Interpreter" > then select the virual environment you just created (says recommended next to it).

## Installing required packages (paste into terminal):
pip install -r requirements.txt

## Interfacing with Github

### Getting the Files from Github
The goal will be to get the files from GitHub onto your machine so you can code in VS Code, and then "push" them back to GitHub once you are done.

1. Open VS Code
2. Press: Ctrl+J
3. Click: terminal
4. type: pwd

pwd = print working directory. This should print the file path in which you are "executing" files at the moment. When you execute the next instruction, you will copy a folder to this location. We usually just keep the working directory as "Desktop" to keep things simple. If you are in a long file path, you can enter "cd .." to move you one folder back. Alternatively, you can write "cd {relative_path_here}" if you know what relative path you want to go to.

Once in the directory you want to remain, run:
git clone https://github.com/CaidenBonney/ME423RoboticsProject

This "cloned" or copied the current GitHub repository into your Desktop. To open it (in terminal), run:
cd ME423RoboticsProject/  

To open it in VS Code (so you can see the files on the left side), click:
file > open folder > Location_Of_Folder_On_Desktop

### Editing the Files
Currently, you are in the "main" branch. As of writing this text, there are two branches, "main" and "cam".  If you wish to switch branches, type into the terminal:
git switch cam
Where "cam" is the name of the branch you wish to be in.

Now you can edit files. Once you are done making edits, navigate to the USB-symbol-looking tab on the left toolbar (Source Control). Here you can see all your edits. At the top, enter a descriptive comment for your changes, click the big blue commit button, and click "Yes" to stage and commit the changes. After this, click the big blue button saying "Sync Changes" and "OK". And you are done

### File Control
If you have had the files open for a while and you want to refresh what you have, you can enter into the terminal:
git pull origin NameOfBranch
where NameOfBranch is the name of the branch (case-sensitive)

If you accidentally, mess up syncing (happens), uhhhhh call us




