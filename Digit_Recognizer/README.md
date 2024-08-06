# MNIST Digit classifier

Ranking for kaggle competition: 20/1801
Accuracy: 100%

MNIST data for classifying digits. My first attempt, I created a neural network. I took one of the CNN's I had built in the past to see how well it would perform. It achieved a 97.635% accuracy which wasn't great

My second attempt I took inspiration from someone elses model but it also didn't perform too well. It achieved a 98.125% accuracy.

My third attempt was something I stumbled across online, I thought it was interesting so I imported it into my project. It was fun to try and figure as it coded everything from scratch including parts of the math. I didn't submit this one as it wouldn't perform well.

My fourth attempt I also took inspiration from someone elses approach. They loaded additional data from tensorflows MNIST dataset. So I did the same. I created a simple CNN using pytorch and added a learning rate reducer. I trained it on 25 epochs with a batch size of 100. It took a couple hours to train and achieved 100% accuracy.