from speed_modules.train import train
from speed_modules.preprocess import generate_opflow

import argparse
class SpeedChallenge:
    def main(self, args):
        if args.mode == "train":
            self.train()
        if args.mode == "preprocess":
            self.preprocess()
    
    def train(self):
        train()
    
    def preprocess(self):
        generate_opflow()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument("training_video",
                        help="training video name")
    parser.add_argument("training_file",
                        help="training data file name")
    parser.add_argument("--model", type=str,
                        help="output model name")
    '''
    parser.add_argument("--mode", choices=["train", "preprocess" ''', "predict"'''],
                        help="Train, Test, or predict model")
    '''
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    '''
    args = parser.parse_args()
    print ("Speed Challenge")
    net = SpeedChallenge()
    net.main(args)