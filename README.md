# PokerAI
Reinforcement-Learning AI that learns Texas Hold'em by playing itself.

## Notes:
* USE A VIRTUALENV (`virtualenvwrapper`) is lit. **Python 3**
* `pip install -r requirements.txt`, and don't forget to add your required modules in there as well
* `git push/pull` your shit often
* Work on your own branches, make pull requests to master

## The Algoithm
We instantiate a game with four instances of our model initiallized with random starting parameters. We run them through an "episode" of poker and pick the best one (i.e. the one with the highest chip stack). An episode of poker is defined as a collection of poker games, which are composed of rounds. A round is defined as one phase of a typical Hold'em game (i.e flop, river, etc). We take the most fit model from each episode and have it play against three clones of itself. We know that each instance of a model will differ in how it predicts since predictions naturally diverge due to the stochastic nature of Q-Learning. We then repeat this step to achieve a pseudo genetic learning algorithm through discounted future prediction.

## PyPokerEngine
For the sake of time, we retrofitted our algorithm to use the pre-coded PyPokerEngine library.
