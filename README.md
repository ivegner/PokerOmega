## 
## A
## The Algorithm
We instantiate a game with four instances of our model initiallized with random starting parameters. We simulate multiple games of poker, with 4 players, each an instance of itself. We know that each instance of a model will differ in how it predicts opponents' moves since predictions naturally diverge due to the stochastic nature of Q-Learning. We then repeat this step to achieve a pseudo genetic learning algorithm through discounted future prediction.

## PyPokerEngine
For the sake of time, we retrofitted our algorithm to use the pre-coded PyPokerEngine library.

## Notes:
* USE A VIRTUALENV (`virtualenvwrapper` is lit). **Python 3**
* `pip install -r requirements.txt`, and don't forget to add your required modules in there as well
* `git push/pull` your shit often
* Work on your own branches, make pull requests to master
