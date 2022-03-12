import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    tasklist = [["push1"], ["push1", "push2"], ["push1", "push2", "cut1"],\
                      ["push1", "push2", "cut1", "cut2"], ["push1", "push2", "cut1", "cut2", "scoop1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1"],\
                      ["push1", "push2", "cut1", "cut2", "scoop1", "scoop2", "open1", "open2"],
                      ["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
                      ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]
    # tasklist = [["open2"], ["open2", "open1"], ["open2", "open1", "scoop2"],\
    #                   ["open2", "open1", "scoop2", "scoop1"], ["open2", "open1", "scoop2", "scoop1", "cut2"],\
    #                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1"],\
    #                   ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2"],
                      # ["open2", "open1", "scoop2", "scoop1", "cut2", "cut1", "push2", "push1"]]

    optimal_rewards = pickle.load(open("optimal_rewards.pkl", "rb"))

    method_rewards = pickle.load(open("rewards_old.pkl", "rb"))
    # print(method_rewards)
    normalized_regret = {}
    model_names = []

    for tasks in tasklist:
        # print(tasks)
        model_name = "_".join(tasks)
        model_names.append(model_name)
        rewards_per_model = method_rewards[model_name]
        # print(rewards_per_model)
        human_only_rewards = [optimal_rewards[task] for task in tasks]
        normalized_regret[model_name] = float(np.mean([(a_i - b_i)/abs(a_i) \
                    for a_i, b_i in zip(human_only_rewards, rewards_per_model)]))

    sorted_regrets = [normalized_regret[model_name] for model_name in model_names]
    print(sorted_regrets)

    # fig, axs = plt.subplots(2,1)

    # axs[0].bar(np.arange(8), sorted_regrets[:8])
    # axs[1].bar(np.arange(len(sorted_regrets[8:])), sorted_regrets[8:])
    # axs[0].set_xticks(np.arange(8))
    # axs[0].set_xticklabels(model_names[:8])
    # axs[1].set_xticks(np.arange(len(sorted_regrets[8:])))
    # axs[1].set_xticklabels(model_names[8:])
    # for ax in axs:
    #     ax.set_ylabel('human_only_rewards - method_rewards')

    # plt.show()
    # method_rewards = {}
    # folder = "runs/no_steptime"
    # rewards_per_model = {}    
    # for filename in os.listdir(folder):
    #     run = pickle.load(open(folder + "/" + filename, "rb"))
    #     rewards_per_model["model"] = run["model"]
    #     rewards_per_model["task"] = run["task"]
    #     rewards_per_model["reward"] = 





if __name__ == "__main__":
    main()