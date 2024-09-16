
# Stock Price Prediction and Analysis

This project is an exploration of various algorithms and tools to analyze and predict stock prices. It utilizes a blend of classic statistical methods, machine learning models, and deep learning techniques. The application is built using Flask to provide a web interface for interaction.

## Tools and Libraries Used

- **Flask**: A lightweight WSGI web application framework in Python. It is used to provide a simple web interface for interacting with the prediction models.
- **YFinance**: A Python library that allows easy access to Yahoo Finance data, which is used to fetch historical stock price data.
- **Moving Average**: A simple yet effective statistical method for smoothing out short-term fluctuations and highlighting longer-term trends in stock prices.
- **RSI (Relative Strength Index)**: A momentum oscillator that measures the speed and change of price movements, commonly used for identifying overbought or oversold conditions.
- **Bollinger Bands**: A volatility indicator consisting of three bands (upper, middle, and lower) that helps in identifying high and low points in the market.
- **Linear Regression**: A basic predictive modeling technique that assumes a linear relationship between the input variables and the output.
- **ARIMA (AutoRegressive Integrated Moving Average)**: A powerful statistical model used for time series forecasting that captures various aspects of a time series, like trend and seasonality.
- **Prophet**: A forecasting tool developed by Facebook that is particularly effective for time series data with daily observations that display seasonality effects.
- **XGBoost**: An advanced gradient boosting algorithm that is highly efficient and often used in winning machine learning competitions.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network particularly suited for modeling sequences and time series data.
- **Random Forest**: An ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.

## Getting Started

To try out the stock price prediction models:

Try It with GitHub Codespaces
You can quickly get started with the stock price prediction models using GitHub Codespaces. Follow these steps:

1. **Log in to GitHub**: Ensure you are logged into your GitHub account.

2. **Access the Repository**: Navigate to the repository on GitHub.

3. **Create a Codespace**: Click on the "Code" button located at the top right of the code window, then select "Create codespace on main".

4. **Activate the Virtual Environment**: Once the Codespace is ready, run the following commands in the terminal:

   ```bash
      pipenv shell
   ```

5. **Install Dependencies**: Install the necessary dependencies by running:

   ```bash
      pip install -r requirements.txt
   ```

6. **Run the Application**: Finally, execute the application with:

```bash
   python app.py
   ```

Local Setup
If you prefer to set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Kishoraditya/stonks.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd stonks
   ```

3. **Start a Virtual Environment**:

   ```bash
   pipenv shell
   ```

4. **Install the Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:

   ```bash
   python app.py
   ```

This README provides a comprehensive overview of the project, including the tools and methods used, as well as instructions for getting started. The "Next Steps" section suggests advanced algorithms and methods that can be explored to further enhance the project's capabilities.
