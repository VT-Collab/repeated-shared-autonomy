from gen_dataset import gen_dataset
from add_noise import add_noise
from train_cae import train_cae
from train_classifier import train_classifier
import sys

def main():
    max_goals = sys.argv[1]
    max_models = 20
    for iter in range (int(max_goals)):
        gen_dataset(iter+1)
        print("dataset generated for number of goals: ", iter+1)
        add_noise()
        for model_no in range(1, max_models+1):
            print("Training cae for number of  goals: ", iter+1)
            train_cae(iter+1, model_no)
            print("Training classifier for number of  goals: ", iter+1)
            train_classifier(iter+1, model_no)


if __name__ == "__main__":
    main()