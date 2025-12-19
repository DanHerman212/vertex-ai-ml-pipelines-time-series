from neuralforecast import NeuralForecast
import pandas as pd

print("Checking NeuralForecast methods...")
print(f"Has predict_insample: {hasattr(NeuralForecast, 'predict_insample')}")
print(f"Has cross_validation: {hasattr(NeuralForecast, 'cross_validation')}")
