# Cryppie-AI-Agent

Cryppie-AI-Agent is an AI agent that uses Hugging Face language models and the Coingecko API to suggest cryptocurrencies according to a user's risk profile.

## Features

- Utilizes Hugging Face language models for natural language processing.
- Integrates with the Coingecko API to fetch real-time cryptocurrency data.
- Suggests cryptocurrencies based on user-defined risk profiles.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ragulcv/Cryppie-AI-Agent.git
    cd Cryppie-AI-Agent
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Configure your API keys in the `config.py` file:
    ```python
    COINGECKO_API_KEY = 'your_coingecko_api_key'
    HUGGING_FACE_API_KEY = 'your_hugging_face_api_key'
    ```

2. Run the main script to start the AI agent:
    ```sh
    python agentcode.py
    ```

