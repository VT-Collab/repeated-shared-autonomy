# repeated-shared-autonomy
project on shared autonomy with repeated interactions

## Simple 2D world

* provided many demonstrations of moving to blue goal and green goal
* learned a policy where each interaction was labeled (either 0 or 1)
* learned a policy where the interactions are not labeled, and the encoder must figure out what the user is currently trying to do
* trained a classifier to determine how confident the model is

To run:
- record_demos.py collects a user demonstration
- process_demos.py converts a demonstration to state - action pairs and adds fake data for the classifier
- train_model_cae.py learns an encoder / decoder model (conditional auto encoder)
- train_model_variation.py learns an encoder / decoder model (conditional variational auto encoder)
- train_classifier.py learns to output confidence given current state
- test_model_variation.py takes joystick inputs and tries to assist using only encoder/decoder
- test_final_setup.py takes joystick inputs and tries to assist using classifier and encoder/decoder
