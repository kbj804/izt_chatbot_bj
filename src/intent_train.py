import os
import sys
path = os.getcwd()
sys.path.append(path)
from src.intent.train import train_intent

if __name__ == '__main__':
    train_intent()