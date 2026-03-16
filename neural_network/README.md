# Predictive Maintenance using Neural Networks

This project predicts whether a printer will require maintenance within the next 30 days. It uses alert logs and job statistics derived from a predictive maintenance dataset.

## Approach and Data Preprocessing
1. **Data Loading and Cleaning**: The Parquet dataset is loaded. Non-predictive identifiers (`PrinterID` and `ReferenceDate`) are removed.
2. **Feature vs Target**: Data is separated into the features (`X`) and the target variable (`y` = `ServiceRequiredNext_30_Days`).
3. **Train-Test Split**: The data is partitioned into training and testing sets with a test size of 20%, applying stratification to maintain class distribution.
4. **SMOTE**: Because predictive maintenance datasets generally suffer from significant class imbalances, Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training set to prevent the model from leaning towards the majority class.
5. **StandardScaler**: Neural networks operate most optimally on scaled numerical inputs. Thus, Standard Scaling resolves magnitude disparities among different features.
6. **Data Reshaping**: For sequence-based neural networks (RNN and LSTM), the 2D scaled data structure is reshaped into a 3D matrix expected by the recurrent layers -> `(batch_size, timesteps=1, features)`.

## Applied Models

Three progressive configurations of neural network models have been applied and trained on the dataset:

1. **Feed Forward Neural Network (FFNN)**
   - The foundational dense network. It includes fully connected layers (`Dense(64)`, `Dense(32)`, `Dense(16)`) coupled with `Dropout(0.3)` layers intended to reduce parameter overfitting.
   
2. **Recurrent Neural Network (RNN)**
   - Employs a `SimpleRNN` architectural node. It attempts to decipher basic short sequence mappings existing within the inputs.

3. **Long Short-Term Memory (LSTM)**
   - Constructed specifically employing `LSTM` cells. This aims to counter shortcomings present in the standard RNN by introducing a memory gating mechanism. Generally, it processes sequential data and vanishing gradient challenges much more adaptively.

## Model Evaluation and Results Comparison

All models output a single unit passing through a `sigmoid` activation function yielding a probability to be converted to binary outputs (>0.5 threshold). They were optimized utilizing `adam` and minimized against `binary_crossentropy` loss. Upon concluding evaluations, the resulting comparison across fundamental classification metrics is displayed below:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Feed Forward NN** | 0.818636 | 0.380325 | 0.444402 | 0.409874 |
| **RNN** | 0.791664 | 0.344531 | 0.520754 | 0.414697 |
| **LSTM** | 0.820231 | 0.385059 | 0.449609 | 0.414838 |

### Observation
- The **RNN** managed to produce the highest *Recall* (0.5207), which indicates it was slightly better at correctly catching the actual positive maintenance triggers compared to the others, though it sacrificed some *Precision* to achieve this.
- Both the **FFNN** and the **LSTM** performed similarly well in *Accuracy* (~82%) and *Precision* (~38%).
- As shown by the closely matched *F1-Scores* (~41%), introducing standard recurrent architecture logic resulted in marginal adjustments to predictive thresholds instead of massive performance overhauls due to the short timeframe structure (`timesteps=1`) applied to the sequence context.
