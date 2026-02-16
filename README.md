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