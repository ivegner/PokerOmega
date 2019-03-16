## PokerOmega:
A self-learning multi-agent poker player, a la AlphaGo. It learns but doesn't work too hot, probably because of insufficient training time/resources.

## Notes:
* USE A VIRTUALENV (`virtualenvwrapper` is great). **Python 3**
* `pip install -r requirements.txt`, and don't forget to add your required modules in there as well
* `git push/pull` often
* Work on your own branches, make pull requests to master

## How to run the code
* python main.py 
* install all the relevant modules with pip install "module name"


## Code details
* Each run is 30 episodes long by default
* Each episode has 100 games by default
* Arguments can be inputted at end of executing the command "python       main.py" with the relevant flags

    * -a (number of agents) - default 4
    * -g (number of games) - default 100
    * -e (number of episodes) - default 30
    * --starte (start epsilon) - default 1.0
    * --edec (epsilon-decay) - default 0.995
    * --emin (epsilon-min) -default 0.01
    * --gamma (gamma) - default(0.95)
    * -o (output file) - default none. If not set, will not save trained model
    * -l (load file) - default none
    * -r (replay-every) - default 20
    * --eval (evaluate every) - default 10
    * --random (evaluate againt random) - default false

