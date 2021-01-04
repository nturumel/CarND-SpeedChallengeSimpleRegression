from speed_modules.train import train
from speed_modules.preprocess import generate_opflow
from globals import * 
import  argparse

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
    initialise()    
    parser = argparse.ArgumentParser()
    #TODO: Add predict
    #TODO: Add resume and wipe 
    parser.add_argument("--mode", choices=["train", "preprocess"],
                        help="Train, Test, or predict model")
    
    def divide_chunks(l, n): 
    # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n]

    parser.add_argument("--set_var_num", nargs='*', help="Pass in args and value", required=False)
    args = parser.parse_args()
    if args.set_var_num is not None:
        x = list(divide_chunks(args.set_var_str, 2)) 
        set_var(x[0], (x[1]))
    
    
    parser.add_argument("--set_var_str", nargs='*', help="Pass in args and value", required=False)
    args = parser.parse_args()
    if args.set_var_str is not None:
        x = list(divide_chunks(args.set_var_str, 2)) 
        set_var(x[0], (x[1]))


    print ("Speed Challenge")
    net = SpeedChallenge()
    net.main(args)