# About
![video](https://github.com/user-attachments/assets/1e1bec25-cbab-471a-9d44-ac399b4c8b24.mp4)

## File Tree

| File     | Description     | 
|--------------|--------------|
| before_training.py | Agent's random behaviour before training. | 
| after_training.py | Agent's behaviour after training |
| notebook.ipynb | Jupyter Notebook giving a breakdown of weather/transfer of heat equation/deep learning | 
| settings.py | Imports python packages & contains global variables | 

# Install
## 1. Clone repository
I recommend clone all repositories into a folder called 'Git' in your C drive. If you don't like the CLI (Command Line Interface), GitHub Desktop is a much friendlier option.

## 2. Setup a virtual environment
These notebooks are not the heaviest in the world, but to be safe, always create a `venv`. Ensure you are in the cloned repository's directory, and then in the CLI:
```
python -m venv [name_of_venv]
```
Activate `venv`:
```
[name_of_venv]\Scripts\activate
```

## 3. Install dependencies
```
pip install -r requirements.txt
```
This will take 10-15 minutes so have a cuppa in the meantime.

# Quick Start
## 1. Run before_training.py
Open VSCode. Open your cloned repository. Open a new terminal and activate your `venv` in the terminal by:
```
[name_of_venv]\Scripts\activate
```
Run file
```
python before_training.py
```
A pygame window will open, showcasing the room's internal temperature throughout the day, when the agent is randomly controlling the heater. To learn more about the environment, see `env.py` file. If the framerate is too fast/slow, open `env.py` and edit the metadata within the RoomSimulator(Env) class.

## 2. Run after_training.py
There are many pre-trained models to load - see `V1_outputs` and `V2_outputs` directories. The number in the filename signifies how many episodes the model trained for so `policy_network_180.keras` means this model trained for 180 episodes. Models trained for more episodes tend to perform better in this environment.

Specify which model you'd like to load in the `after_training.py` file, where the `policy_network` variable is defined. Again, in the terminal run:
```
python after_training.py
```

# More Information
More information on how the weather file and environment was constructed can be found in the `notebook.ipynb`.
