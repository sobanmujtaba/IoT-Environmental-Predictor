# IoT Environmental Predictor

Regression model that predicts indoor temperature from IoT sensor readings — temperature (DHT11), humidity (DHT11), light intensity (LDR), and CO2 concentration.

Built with Python, scikit-learn, and the [UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy+detection).

**[Live demo and explanation](https://sobanmujtaba.github.io/IoT-Environmental-Predictor)**

---

## What it does

Takes readings from common IoT sensors and learns to predict indoor temperature. Three regression models are compared across two separate test windows to evaluate generalization.

| Model              | Test1 R2 | Test1 RMSE | Test2 R2 | Test2 RMSE |
|--------------------|----------|------------|----------|------------|
| Linear Regression  | 0.9716   | 0.1733 °C  | 0.8981   | 0.3258 °C  |
| Random Forest      | 0.1155   | 0.9666 °C  | -0.7905  | 1.3657 °C  |
| Gradient Boosting  | 0.3936   | 0.8004 °C  | 0.1215   | 0.9566 °C  |

Linear regression wins. The tree models overfit to training-week thermal patterns and break on a different week — a real-world distribution shift problem in IoT data.

---

## Dataset

UCI Occupancy Detection — real office sensors, 1-minute intervals, February 2015.

Download the three files and place them alongside `predict.py`:
- `datatraining.txt`
- `datatest.txt`
- `datatest2.txt`

---

## Run

```bash
pip install scikit-learn pandas numpy matplotlib
python predict.py
```

---

## Files

```
predict.py        main script
assets/           EDA and evaluation plots
index.html        GitHub Pages explainer
```

---

## Key finding

Light intensity (LDR) is the strongest predictor of temperature in an occupied office — more than humidity itself. Lights on means people are in the room, and people generate heat. The LDR reading ends up encoding occupancy better than CO2 in this dataset.
