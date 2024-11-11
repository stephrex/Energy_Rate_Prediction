import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import joblib

# Function to load the models
class NBeatsBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_layers=4, num_neurons=512, **kwargs):
        super(NBeatsBlock, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.hidden_layers = [layers.Dense(
            self.num_neurons, activation='relu') for _ in range(self.num_layers)]
        self.theta_layer = layers.Dense(
            self.output_dim + self.input_dim, activation='linear')

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:,
                                   :self.input_dim], theta[:, self.input_dim:]
        return backcast, forecast

    def get_config(self):
        config = super(NBeatsBlock, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_neurons': self.num_neurons
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NBeats(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_blocks=3, **kwargs):
        super(NBeats, self).__init__(**kwargs)
        self.blocks = [NBeatsBlock(input_dim, output_dim)
                       for _ in range(num_blocks)]

    def call(self, x):
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast
            forecast += block_forecast
        return forecast

    def get_config(self):
        config = super(NBeats, self).get_config()
        config.update({
            'input_dim': self.blocks[0].input_dim,
            'output_dim': self.blocks[0].output_dim,
            'num_blocks': len(self.blocks)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, safe_mode=False, custom_objects={'NBeats': NBeats, 'NBeatsBlock': NBeatsBlock})

scaler1_x_path = 'Scalers/scaler1_x.pkl'
scaler1_y_path = 'Scalers/scaler1.pkl'
scaler2_x_path = 'Scalers/scaler2_x.pkl'
scaler2_y_path = 'Scalers/scaler2.pkl'
scaler3_x_path = 'Scalers/scaler3_x.pkl'
scaler3_y_path = 'Scalers/scaler3.pkl'

# Load models for each submeter
model_submeter1_path = 'models/004_LSTM_SubMeter1.keras'
model_submeter2_path = 'models/004_LSTM_SubMeter1.keras'
model_submeter3_path = 'models/004_LSTM_SubMeter1.keras'


# Function to make predictions based on the input values and selected model
def make_predictions(model_path, active_power, voltage, intensity, submeter_value, scaler_x_path, scaler_y_path):
    model = load_model(model_path)
    # Prepare input for the model (reshape and concatenate features appropriately)
    active_power = np.array(active_power).flatten()
    voltage = np.array([voltage]).flatten()
    intensity = np.array([intensity]).flatten()
    submeter_value = np.array([submeter_value]).flatten()
    
    features = np.concatenate([active_power, voltage, intensity, submeter_value])
    features = tf.expand_dims(features, axis=0)

    scaler_x = load_scaler(scaler_x_path)
    scaler_y = load_scaler(scaler_y_path)

    scaled_features = scaler_x.transform(features)
    predictions = model.predict(scaled_features)
    predictions = scaler_y.inverse_transform(predictions)
    predictions = np.maximum(predictions, 0)

    return tf.squeeze(predictions)

def main():
    st.set_page_config(
        page_title="Power Usage Prediction System", 
        page_icon="⚡", 
        layout="wide", 
        initial_sidebar_state="expanded",
    )

    st.title("⚡ Power Usage Prediction System")
    st.markdown("""
        This app predicts power usage over a two-week period based on daily inputs of Active Power, Voltage, Intensity, and Submeter readings.
    """, unsafe_allow_html=True)

    # Sidebar Styling
    st.sidebar.header("Configuration")
    st.sidebar.markdown("⚙️ **Customize Your Inputs Below**", unsafe_allow_html=True)

    # Submeter Selection
    submeter_option = st.sidebar.selectbox(
        "Select Submeter Category:", ("Submeter 1", "Submeter 2", "Submeter 3"))

    st.header("Enter Active Power Values for 24 Hours")
    cols = st.columns(6)  # Divide into 6 columns for better layout
    active_power = []

    for i in range(24):
        col = cols[i % 6]  # Use modulo to distribute inputs across columns
        with col:
            value = st.number_input(f"Hour {i+1}", min_value=0.0, value=0.0, step=0.1, key=f"hour_{i+1}")
            active_power.append(value)

    # Input for Voltage, Intensity, and Submeter value (single values)
    st.markdown("---")
    st.subheader("Voltage, Intensity, and Submeter Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        voltage = st.number_input(
            "Voltage (V):", value=230.0, step=0.1, format="%.1f")
    with col2:
        intensity = st.number_input(
            "Intensity (A):", value=5.0, step=0.1, format="%.1f")
    with col3:
        submeter_value = st.number_input(
            f"{submeter_option} Value (Wh):", value=0.0, step=0.1, format="%.1f")

    # Load the correct model based on the selected submeter
    if submeter_option == "Submeter 1":
      scaler_x_path = scaler1_x_path
      scaler_y_path = scaler1_y_path
      selected_model_path = model_submeter1_path
    elif submeter_option == "Submeter 2":
      scaler_x_path = scaler2_x_path
      scaler_y_path = scaler2_y_path
      selected_model_path = model_submeter2_path
    else:
      scaler_x_path = scaler3_x_path
      scaler_y_path = scaler3_y_path
      selected_model_path = model_submeter3_path

    # Make predictions
    if st.button("⚡ Predict"):
        with st.spinner('Predicting... Please wait.'):
            predictions = make_predictions(
                selected_model_path, active_power, voltage, intensity, submeter_value, scaler_x_path, scaler_y_path)

        # Plotting the predictions
        st.subheader("Predicted Power Usage for the Next 2 Weeks")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(predictions, label='Predicted Power Usage', color='blue', linewidth=1.2, linestyle='-')
        ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Days', fontsize=12, fontweight='bold', color='navy')
        ax.set_ylabel('Power Usage (kW)', fontsize=12, fontweight='bold', color='navy')
        ax.set_title(f"Power Usage Prediction for {submeter_option} Over 2 Weeks", fontsize=14, fontweight='bold', color='darkred')
        ax.set_xticks(np.arange(0, len(predictions), 24))
        ax.set_xticklabels([f'Day {i+1}' for i in range(len(predictions) // 24)], rotation=45, fontsize=10)
        for i in range(0, len(predictions)//24):
            ax.axvspan(i*24, i*24 + 24, color='lightgrey', alpha=0.1 if i % 2 == 0 else 0.15)
        ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, facecolor='lightyellow')
        peak_index = np.argmax(predictions)
        peak_value = predictions[peak_index]
        ax.plot(peak_index, peak_value, marker='o', color='red', markersize=8)
        ax.text(peak_index + 2, peak_value, f'Peak: {peak_value:.2f} kW', 
                ha='left', fontsize=10, color='darkred', fontweight='bold')
        st.pyplot(fig)



if __name__ == "__main__":
    main()