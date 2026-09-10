We have made additional changes to improve the robustness of the results. The code does not reproduce the figures from the paper one-to-one.

## Post-publication refinement

Subsequent validation across different IQP architectures and tested values of β demonstrates the practical value of parity supervision as an inductive bias, with benefits governed by the architecture and target distribution. Within suitable tested regimes, replacing MSE with a parity-moment loss improves generalization while keeping the circuit architecture fixed, without requiring additional qubits or greater circuit depth. These results highlight the importance of aligning circuit expressivity, the training objective and the target’s spectral structure. The choice of supervision therefore provides a practical, additional way to improve an existing IQP model.
