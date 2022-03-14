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

    optimal_rewards = pickle.load(open("optimal_rewards.pkl", "rb"))

    files = ["rewards_old_alpha_0.8_runs_5.pkl", "rewards_old_alpha_nolimit_runs_5.pkl"]
    total_regrets = []
    for file in files:
        method_rewards = pickle.load(open(file, "rb"))
        
        tasklist = tasklist[:len(method_rewards)]
        normalized_regret = {}
        model_names = []

        for tasks in tasklist:
            # print(tasks)
            model_name = "_".join(tasks)
            model_names.append(model_name)
            rewards_per_model = method_rewards[model_name]
            # print(rewards_per_model)
            human_only_rewards = [optimal_rewards[task][0] for task in tasks]
            normalized_regret[model_name] = float(np.mean([(a_i - b_i)/abs(a_i) \
                        for a_i, b_i in zip(human_only_rewards, rewards_per_model)]))
            if model_name == "open2":
                print(optimal_rewards["open2"], method_rewards["open2"], normalized_regret["open2"])
        sorted_regrets = [normalized_regret[model_name] for model_name in model_names]
        total_regrets.append(sorted_regrets)
        # if len(sorted_regrets) <= 8:
        #     print(sorted_regrets)
        # else:
        #     print(sorted_regrets[:8])
        #     print(sorted_regrets[8:])
    fig, axs = plt.subplots(2,1)
    x = np.arange(1, 9)
    width = 0.35
    start = 0
    end = 8
    for ax in axs.ravel():
        ax.bar(x - width/2, total_regrets[0][start:end], width, label="fixed blending")
        ax.bar(x + width/2, total_regrets[1][start:end], width, label="alpha = [0,0.6]")
        ax.set_ylabel("normalized_regret")
        start += 8
        end += 8
    ax.set_xlabel("skills learned")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.suptitle("regret vs models")
    plt.show()


if __name__ == "__main__":
    main()