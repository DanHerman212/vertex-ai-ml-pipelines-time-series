# Model Evaluation Report

This document presents the evaluation results for the two deep learning architectures trained for subway headway prediction: **GRU (Gated Recurrent Unit)** and **N-HiTS (Neural Hierarchical Interpolation for Time Series)**.

## 1. GRU Model Evaluation

The GRU model is a Recurrent Neural Network (RNN) designed to capture temporal dependencies in sequential data. It serves as our baseline deep learning model.

<p align="center">
  <img src="../images/gru_model_evaluation.png" alt="GRU Model Evaluation" width="800">
</p>

### Interpretation
*   **Prediction Accuracy**: The prediction plot compares the model's forecasted headway (Minutes Between Trains) against the actual ground truth. The GRU model generally captures the main trend of the data but may struggle with high-frequency noise or sudden spikes in delays.
*   **Lag**: RNNs can sometimes exhibit a "lag" behavior where the prediction is simply a shifted version of the previous time step. The evaluation metrics (MAE/MSE) help quantify this behavior.

---

## 2. N-HiTS Model Evaluation

N-HiTS is a state-of-the-art MLP-based architecture that uses hierarchical interpolation to capture time series patterns at multiple scales (e.g., long-term trends vs. short-term fluctuations).

<p align="center">
  <img src="../images/nhits_model_evaluation.png" alt="N-HiTS Model Evaluation" width="800">
</p>

### Interpretation
*   **Multi-Scale Forecasting**: N-HiTS typically excels at capturing both the daily seasonality (rush hours vs. late night) and the immediate short-term dependencies.
*   **Sharpness**: Compared to the GRU, the N-HiTS predictions often appear "sharper" and more responsive to sudden changes in headway, thanks to its hierarchical blocks.

---

## 3. Model Comparison

| Feature | GRU (Baseline) | N-HiTS (Advanced) |
| :--- | :--- | :--- |
| **Architecture** | Recurrent (Sequential) | MLP (Hierarchical) |
| **Training Speed** | Slower (Sequential processing) | Faster (Parallelizable) |
| **Long-Term Memory** | Limited by vanishing gradient | Excellent (Hierarchical sampling) |
| **Performance** | Good for short horizons | Superior for long horizons & complex seasonality |

### Conclusion
Based on the evaluation plots and metrics, the **N-HiTS model** is selected as the primary candidate for production deployment. It offers a better balance of accuracy and computational efficiency, particularly for the variable nature of subway arrival times.
