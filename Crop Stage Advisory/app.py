import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy

# Load the trained model
model_path = 'xgb_model.pkl'
try:
    with open(model_path, 'rb') as model_file:
        xgb_model = pickle.load(model_file)
        st.success('Model loaded successfully!')
except Exception as e:
    st.error(f'An error occurred while loading the model: {e}')
    xgb_model = None

# Load the processed dataset
data = pd.read_csv('Processed_Crop_stage.csv')

# Convert 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# Define advisories for each crop stage
advisories = {
    "Establishment stage": [
        "Ensure proper spacing and planting depth.",
        "Maintain adequate soil moisture. Avoid waterlogging.",
        "Begin initial weeding to minimize competition for nutrients.",
        "Apply initial doses of fertilizers and organic matter."
    ],
    "Vegetative stage": [
        "Apply balanced fertilizers.",
        "Regularly water the crops but avoid overwatering.",
        "Monitor for pests and diseases.",
        "Continue regular weeding."
    ],
    "Shooting stage": [
        "Provide physical support to crops if needed.",
        "Be vigilant for signs of pests and diseases.",
        "Apply nutrients that support flowering and fruiting.",
        "Maintain consistent watering."
    ],
    "Development and harvesting stage": [
        "Regularly check the maturity of crops.",
        "Continue to monitor for pests.",
        "Prepare for harvesting by ensuring storage facilities are clean.",
        "Handle crops carefully during harvesting."
    ]
}

# Streamlit UI
st.set_page_config(page_title="Crop Stage Advisory", page_icon="🌱", layout="wide")

# Sidebar
with st.sidebar:
    st.image("C:/Users/Hp/Desktop/2/Logo.jpg", width=100)  # Add logo to the sidebar
    st.header("Select Parameters")

    state = st.selectbox("Select State", options=data['State'].unique())
    district = st.selectbox("Select District", options=data[data['State'] == state]['District Name'].unique())
    commodity = st.selectbox("Select Commodity", options=data[data['District Name'] == district]['Commodity'].unique())
    selected_date = st.date_input("Select Date", value=datetime.date.today())

# Main content
st.title("🌾 Crop Stage Advisory 🌾")
st.markdown("---")
st.header("Selected Parameters")
st.write(f"**State:** {state}")
st.write(f"**District:** {district}")
st.write(f"**Commodity:** {commodity}")
st.write(f"**Date:** {selected_date}")

# Add a Predict button
if st.button('Predict'):
    # Filter data based on the selected inputs
    filtered_data = data[(data['State'] == state) &
                          (data['District Name'] == district) &
                          (data['Commodity'] == commodity) &
                          (data['Date'] == pd.to_datetime(selected_date))]

    if not filtered_data.empty:
        crop_stage = filtered_data.iloc[0]['Crop stage growth']
        st.subheader(f"Crop Stage: {crop_stage}")

        # Display advisories
        if crop_stage in advisories:
            st.markdown("### Advisories")
            for advice in advisories[crop_stage]:
                st.write(f"🌱 {advice}")
        else:
            st.write("No advisories available for this crop stage.")

        # Display weather parameters
        st.markdown("### Weather Parameters")
        st.write(f"**Temperature Max:** {filtered_data.iloc[0]['TempMax']} °C")
        st.write(f"**Temperature Min:** {filtered_data.iloc[0]['TempMin']} °C")
        st.write(f"**Humidity:** {filtered_data.iloc[0]['Humidity']} %")
        st.write(f"**Precipitation:** {filtered_data.iloc[0]['Precipitation']} mm")
        st.write(f"**Windspeed:** {filtered_data.iloc[0]['Windspeed']} km/h")
        st.write(f"**Solar Radiation:** {filtered_data.iloc[0]['SolarRadiation']} MJ/m²")

        # Data Visualization
        st.markdown("### Weather Trends (Next 15 Days)")

        # Update date range to start from the selected date
        date_range = pd.date_range(start=selected_date, periods=15)

        # Generate example weather forecast data for the next 15 days starting from selected_date
        weather_data = {
            "Date": date_range,
            "Temperature Max (°C)": np.random.uniform(25, 35, size=15),
            "Temperature Min (°C)": np.random.uniform(15, 25, size=15),
            "Humidity (%)": np.random.uniform(60, 90, size=15),
            "Precipitation (mm)": np.random.uniform(0, 10, size=15),
            "Windspeed (km/h)": np.random.uniform(5, 20, size=15),
            "Solar Radiation (MJ/m²)": np.random.uniform(150, 250, size=15)
        }
        weather_df = pd.DataFrame(weather_data)

        # Create subplots for weather data
        fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        fig.suptitle('Weather Trends for Next 15 Days', fontsize=16)

        # Max Temperature
        axs[0, 0].plot(weather_df['Date'], weather_df['Temperature Max (°C)'], color='red')
        axs[0, 0].set_title('Max Temperature')
        axs[0, 0].set_ylabel('Temperature (°C)')

        # Min Temperature
        axs[0, 1].plot(weather_df['Date'], weather_df['Temperature Min (°C)'], color='blue')
        axs[0, 1].set_title('Min Temperature')
        axs[0, 1].set_ylabel('Temperature (°C)')

        # Humidity
        axs[1, 0].plot(weather_df['Date'], weather_df['Humidity (%)'], color='green')
        axs[1, 0].set_title('Humidity')
        axs[1, 0].set_ylabel('Humidity (%)')

        # Precipitation
        axs[1, 1].plot(weather_df['Date'], weather_df['Precipitation (mm)'], color='purple')
        axs[1, 1].set_title('Precipitation')
        axs[1, 1].set_ylabel('Precipitation (mm)')

        # Windspeed
        axs[2, 0].plot(-weather_df['Date'], weather_df['Windspeed (km/h)'], color='orange')
        axs[2, 0].set_title('Windspeed')
        axs[2, 0].set_ylabel('Windspeed (km/h)')

        # Solar Radiation
        axs[2, 1].plot(weather_df['Date'], weather_df['Solar Radiation (MJ/m²)'], color='brown')
        axs[2, 1].set_title('Solar Radiation')
        axs[2, 1].set_ylabel('Solar Radiation (MJ/m²)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

    else:
        st.write("No data available for the selected inputs.")
