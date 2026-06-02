# Physics-Informed AI for Dynamic Aircraft Balancing

This repository contains the implementation of a custom deep learning architecture designed to dynamically balance an aircraft using a movable counterweight system. Developed as part of a graduation project under the **TUSAŞ Lift-Up Program**, this model bridges the gap between deep learning predictions and physical mechanical constraints.

##  Overview
While standard neural networks excel at finding mathematical ideals, they lack the awareness of real-world mechanical limits. In this project, an AI model is trained to predict the required position of a movable counterweight to maintain an aircraft's center of gravity. 

The core engineering challenge: **The lead screw motor pushing the weight has a physical speed limit.** A standard LSTM + Attention model often requests the counterweight to move at physically impossible speeds (teleportation). To solve this, we implemented a custom **Physics-Informed Attention** mechanism and a physics-guided loss function.

##  Architecture
The model relies on a time-series forecasting approach using a 10-step sliding window, processed through a custom pipeline:

* **1D CNN (Noise Filter):** Acts as a feature extractor that smooths out momentary sensor noise and engine vibrations before they reach the main network.
* **LSTM (Memory Module):** Processes the cleaned sequence to understand the aircraft's pitching trend (momentum).
* **Physics-Informed Attention:** A custom layer built from scratch in PyTorch. It subtracts a dynamic "Physical Penalty Matrix" from the standard scaled dot-product attention scores. If a mathematical connection requires a physical movement exceeding the motor's speed limit, its probability is dropped to near zero.
* **Speed-Limit Loss:** A custom loss function that heavily penalizes the model during backpropagation if its final predicted position violates the mechanical speed boundaries.
