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


    ideal_folder = "runs/individual_models"
    nolimit_folder = "runs/alpha_nolimit"
    limit_folder = "runs/alpha_0.8"

    optimal_traj = {}
    for filename in os.listdir(ideal_folder):
        if not filename[0] == "m":
            continue
        demo = pickle.load(open(ideal_folder + "/" + filename, "rb"))
        key = demo["task"]
        if key in optimal_traj:
            curr_traj = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
            optimal_traj[key] = np.append(optimal_traj[key], curr_traj, axis=0)
        else:
            optimal_traj[key] = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
    
    for key in optimal_traj:
        optimal_traj[key] = np.mean(optimal_traj[key], axis=0)
    
    nolimit_traj = {}
    limit_traj = {}
    model_names = []
    for taskset in tasklist:
        model_name = "_".join(taskset)
        model_names.append(model_name)
        trajs_nolimit = {}
        trajs_limit = {}
        for filename in os.listdir(nolimit_folder):
            if not filename[0] == "m":
                continue
            demo = pickle.load(open(nolimit_folder + "/" + filename, "rb"))
            if not model_name == demo["model"]:
                continue
            task = demo["task"]
            if task in trajs_nolimit:
                curr_traj = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
                trajs_nolimit[task] = np.append(trajs_nolimit[task], curr_traj, axis=0)
            else:
                trajs_nolimit[task] = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)

            demo = pickle.load(open(limit_folder + "/" + filename, "rb"))
            task = demo["task"]
            if task in trajs_limit:
                curr_traj = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
                trajs_limit[task] = np.append(trajs_limit[task], curr_traj, axis=0)
            else:
                trajs_limit[task] = np.array([item[1] for item in demo["data"]]).reshape(1, len(demo["data"]), 6)
        t_nlimit = []
        t_limit = []
        for task in trajs_nolimit:
            trajs_nolimit[task] = float(np.linalg.norm(np.mean(trajs_nolimit[task] - optimal_traj[task], axis=0)))
            trajs_limit[task] = float(np.linalg.norm(np.mean(trajs_limit[task] - optimal_traj[task], axis=0)))
            t_nlimit.append([trajs_nolimit[task]])
            t_limit.append([trajs_limit[task]])
        # print(t_nlimit)
        nolimit_traj[model_name] = np.mean(t_nlimit)
        limit_traj[model_name] = np.mean(t_limit)

    # print(nolimit_traj)
    # print(limit_traj)
    nolimit = [nolimit_traj[model_name] for model_name in model_names]
    limit = [limit_traj[model_name] for model_name in model_names]
    fig, axs = plt.subplots(2,1)
    x = np.arange(1, 9)
    width = 0.35
    start = 0
    end = 8
    for ax in axs.ravel():
        ax.bar(x - width/2, limit[start:end], width, label="fixed blending")
        ax.bar(x + width/2, nolimit[start:end], width, label="alpha = [0,0.6]")
        ax.set_ylabel("regret = mean dist between ideal traj and model traj")
        start += 8
        end += 8
    ax.set_xlabel("skills learned")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.suptitle("regret vs models")
    plt.show()
    # optimal_rewards = pickle.load(open("optimal_rewards.pkl", "rb"))

    # files = ["rewards_old_alpha_0.8_runs_5.pkl", "rewards_old_alpha_nolimit_runs_5.pkl"]
    # total_regrets = []
    # for file in files:
    #     method_rewards = pickle.load(open(file, "rb"))
        
    #     tasklist = tasklist[:len(method_rewards)]
    #     normalized_regret = {}
    #     model_names = []

    #     for tasks in tasklist:
    #         # print(tasks)
    #         model_name = "_".join(tasks)
    #         model_names.append(model_name)
    #         rewards_per_model = method_rewards[model_name]
    #         # print(rewards_per_model)
    #         human_only_rewards = [optimal_rewards[task][0] for task in tasks]
    #         normalized_regret[model_name] = float(np.mean([(a_i - b_i)/abs(a_i) \
    #                     for a_i, b_i in zip(human_only_rewards, rewards_per_model)]))
    #         if model_name == "open2":
    #             print(optimal_rewards["open2"], method_rewards["open2"], normalized_regret["open2"])
    #     sorted_regrets = [normalized_regret[model_name] for model_name in model_names]
    #     total_regrets.append(sorted_regrets)
        # if len(sorted_regrets) <= 8:
        #     print(sorted_regrets)
        # else:
        #     print(sorted_regrets[:8])
        #     print(sorted_regrets[8:])



if __name__ == "__main__":
    main()