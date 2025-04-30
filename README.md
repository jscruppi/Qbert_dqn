# Qbert DQN
Demo of several different AI packages in python in order to make/train an AI model to play the old Atari 2600 game "Qbert".

# Getting Started
First things first, you'll need a working clone of the repo on your machine

        $ git clone git@github.com/jscruppi/Qbert_dqn 
        $ cd Qbert_dqn        

Once that is complete, a python envirnoment must be setup. The included Pipfile in the repo should have all the correct packages and their versions for proper use. To start this environment, make sure pipenv is installed.

        $ pipenv --version
        pipenv, version 2024.4.0

Note: I have not tested the environment with any other versions of pipenv

If pipenv is not installed, you can use pip to install it as such:

        $ pip install pipenv --user

Once pipenv is properly installed, use it to install all the needed packages

        $ pipenv install

After the installation is complete, use pipenv to run the model:

        $ pipenv run python3 gym.py

