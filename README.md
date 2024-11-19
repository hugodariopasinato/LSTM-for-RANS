# LSTM Turbulence Model Prediction Repository

This repository contains the necessary files and code used for predicting Reynolds stress \(\langle uv \rangle^+\) in statistical 2D turbulent channel flows using a Long Short-Term Memory (LSTM) neural network, as described in the paper 'Modeling turbulent flow with LSTM neural networks', and 'Using LSTM predictions for RANS simulations'.

## Files

1. **`params_4XBL13BL23BL33DL_T32.txt`**  
   This file contains the trained LSTM model parameters. The model was trained on sequences of 32 elements representing flow features.

2. **`maxmin_4XBL13BL23BL33DL.dat`**  
   This file provides the maximum and minimum values for the input features and output predictions. These values are used to normalize and de-normalize the data during prediction.

3. **Fortran Subroutines for Prediction**  
   This repository includes Fortran subroutines that can be incorporated into a CFD code to perform RANS simulations with LSTM-based predictions of Reynolds stress \(\langle uv \rangle^+\). These subroutines load the LSTM model parameters and the normalization data, allowing the CFD solver to predict Reynolds stresses at each timestep as part of the simulation process.

## How to Use

1. **Normalization and De-normalization**  
   The input features should be normalized before passing them to the LSTM model. Use the formula:
   \[
   \text{normalized\_val} = \frac{\text{val} - \text{min}}{\text{max} - \text{min}}
   \]
   The predicted values are also normalized and should be de-normalized using:
   \[
   \text{val} = \text{normalized\_val} \times (\text{max} - \text{min}) + \text{min}
   \]

2. **Features and Predictions**  
   - The input features are sequences of 32 elements with the format:  
     \[
     (Re_{\tau}, S_{11}^+, S_{12}^+, Y)
     \]
   - The predictions are sequences of the normalized Reynolds stress \(\langle uv \rangle^+\).
   - The friction Reynolds number range should be 150-600.
     
3. **Integrating the Fortran Subroutines**  
   - The provided Fortran subroutines should be incorporated into a CFD code to perform RANS simulations.
   - Ensure that the `params_4XBL13BL23BL33DL_T32.txt` and `maxmin_4XBL13BL23BL33DL.dat` files are available to the CFD code.
   - The subroutines load the LSTM model parameters and use the normalized input data to predict Reynolds stresses during each timestep of the simulation.

## Citation

If you use this code or data in your research, please cite the corresponding paper:  

H.D. Pasinato. Modeling turbulent flows with LSTM neural network. arXiv:2307.13784v1[physics.flu-dyn], 2023 [https://arxiv.org/abs/2307.13784]

H.D. Pasinato. Using LSTM Predictions for RANS Simulations [https://arxiv.org/abs/2411.11723].
