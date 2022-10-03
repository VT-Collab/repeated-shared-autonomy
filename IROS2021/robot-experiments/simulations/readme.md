# Data organization for the three tasks

The data is located in "data/consolidated_data/"

## task1.pkl
Task 1: Robot moves autonomously towards intended target after human controls for the first 0.5 seconds.

Contains a dictionary with L2 norm target and final position of robot in catesian space.

    ours = {}
    ours["c"] = Distance from soup can target after 5 demonstrations
    ours["n"] = Distance from notepad target after 5 demonstrations
    ours["t"] = Distance from tape measure target after 5 demonstrations

    bc = {}
    bc["c"] = Distance from soup can target after 5 demonstrations
    bc["n"] = Distance from notepad target after 5 demonstrations
    bc["t"] = Distance from tape measure can target after 5 demonstrations

    dataset = {}
    dataset["5"] = [ours, bc]

    ours = {}
    ours["c"] = Distance from soup can target after 3 demonstrations
    ours["n"] = Distance from notepad target after 3 demonstrations
    ours["t"] = Distance from tape measure target after 3 demonstrations

    bc = {}
    bc["c"] = Distance from soup can target after 3 demonstrations
    bc["n"] = Distance from notepad target after 3 demonstrations
    bc["t"] = Distance from tape measure target after 3 demonstrations

    dataset["3"] = [ours, bc]

## task2.pkl

Task 2: Robot and human work towards known goals.

Contains a list with confidence over time for trajectories.

    ours = {}
    ours["g"] = Confidence while moving towards known goal glass
    ours["n"] = Confidence while moving towards known goal notepad

    dropout = {}
    dropout["g"] = Confidence while moving towards known goal glass
    dropout["n"] = Confidence while moving towards known goal notepad

    dataset = [ours, dropout]


## task3.pkl

Task 3: Comparison between human only and human+robot while reaching towards known and unknown goals. In this task known goal is "s" and unknown goal is "g".

    ours = {}
    ours["s"] = Human effort while moving towards known goal shelf
    ours["g"] = Human effort while moving towards unknown goal glass

    human = {}
    human["s"] = Human effort while moving towards known goal shelf
    human["g"] = Human effort while moving towards unknown goal glass

    dataset = [ours, human]

## task3_noise.pkl

Task 3 noise: Comparison between noisy human only and noisy human + robot while reaching a known goal. Here noise range is {0.01, 0.05, 0.07}.

    ours = {}
    ours["001"] = Human effort with noise = 0.01
    ours["005"] = Human effort with noise = 0.05
    ours["007"] = Human effort with noise = 0.07

    human = {}
    human["001"] = Human effort with noise = 0.01
    human["005"] = Human effort with noise = 0.05
    human["007"] = Human effort with noise = 0.07

    dataset = [ours, human]