<!-- If In VS Code press Ctrl + Shift + V to view this file in a Markdown Preview -->
# Complete Code Description is in README.md in the src folder
[Complete Code Description](./src/README.md)


# Requirements 
### Python Version: 
3.12.10

### Quanser API
The Quanser API is required to run the arm code. If using a virtual environment ensure that the Quanser API is installed within the environment. Set up of a virtual environment is described in the installation instructions below. After creation and activation of the virtual environment, the Quanser API can be installed to this environment by running the Quanser API installer script. 

Note the virutal environment must be activated before running the script and the script must be run from the terminal that is the virtual environment was activated from.


# Installation Instructions
## Windows Installation Instructions:
### Go to Windows Powershell and paste the following:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop --version
scoop install git
scoop bucket add extras
scoop install uv
uv python install 3.12.10
uv python update-shell
```

### Create a virtual environment after opening the project in VS Code by pasting the following into the terminal:
```
uv run --python 3.12.10 python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-win.txt
```

After creating the virtual environment, set the python interpreter to the virtual environment by going to VS Code's Search at the top:
">Python: Select Interpreter" > then select the virual environment you just created (says recommended next to it).

## Mac Installation Instructions:
### Go to Terminal and paste the following:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew update
brew install git
brew install pyenv
pyenv install 3.12.10
pyenv global 3.12.10
```

### Create a virtual environment after opening the project in VS Code by pasting the following into the terminal:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
```

After creating the virtual environment, set the python interpreter to the virtual environment by going to VS Code's Search at the top:
">Python: Select Interpreter" > then select the virual environment you just created (says recommended next to it).