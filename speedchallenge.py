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
    #TODO: Add predict
    #TODO: Add resume and wipe 
    parser.add_argument("--mode", choices=["train", "preprocess"],
                        help="Train, Test, or predict model")
    

    args = parser.parse_args()
    print ("Speed Challenge")
    net = SpeedChallenge()
    net.main(args)