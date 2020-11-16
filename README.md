# repeated-shared-autonomy
project on shared autonomy with repeated interactions

## Simple 2D world

* provided many demonstrations of moving to blue goal and green goal
* learned a policy where each interaction was labeled (either 0 or 1)
* learned a policy where the interactions are not labeled, and the encoder must figure out what the user is currently trying to do

To run:
- record_demos.py collects a user demonstration
- process_demos.py converts a demonstration to state - action pairs
- train_model.py learns an encoder / decoder model
- test_model.py takes joystick inputs and tries to assist
