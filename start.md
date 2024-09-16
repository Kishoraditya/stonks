# Getting Started with Stock Price Prediction and Analysis

This guide provides instructions for setting up and running the Stock Price Prediction and Analysis project using different methods.

## 1. GitHub Codespaces

1. Open the project in GitHub Codespaces.

2. In the terminal, run:

    ```bash
        pip install -r requirements.txt python app.py
    ```

3. Click on the "Open in Browser" button when prompted.

## 2. Local Development

1. Clone the repository:

    ```bash
        git clone cd stock-prediction
    ```

2. Create and activate a virtual environment:

    ```bash
        python -m venv venv source venv/bin/activate # On Windows, use venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
        pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
        python app.py
    ```

5. Open a web browser and navigate to `http://localhost:5000`.

## 3. Using Docker

1. Build the Docker image:

    ```bash
        docker build -t stock-prediction .
    ```

2. Run the Docker container:

    ```bash
        docker run -p 5000:5000 stock-prediction
    ```

3. Open a web browser and navigate to `http://localhost:5000`.

## 4. Using Docker Compose

1. Start the application:

    ```bash
        docker-compose up --build
    ```

2. Open a web browser and navigate to `http://localhost:5000`.

## 5. Deploying with Terraform on AWS

1. Install Terraform and configure AWS credentials.

2. Initialize Terraform:

    ```bash
        terraform init
    ```

3. Review the deployment plan:

    ```bash
        terraform plan -out=tfplan.json -input=false -lock=false -no-color -detailed-exitcode
    ```

4. Apply the Terraform configuration:

    ```bash
        terraform apply "tfplan.json"
    ```

5. Access the application using the EC2 instance's public IP address.

## Additional Notes

- Ensure you have the necessary permissions and credentials for AWS deployment.

- For local development, make sure you have Python 3.9 installed.

- When using Docker or Terraform, ensure Docker is installed on your system.

- Modify the `.env` file with appropriate values before deployment.
