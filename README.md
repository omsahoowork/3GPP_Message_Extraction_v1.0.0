# LangGraph Agentic Deployment

A comprehensive application for deploying agentic workflows using LangGraph with Streamlit.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- An OpenAI or Anthropic API key with a larger context window account

## Setup Instructions

### 1. Create a Python Environment

To create a virtual environment, run the following command in your project directory:

```bash
python -m venv venv
```

This will create a new virtual environment folder named `venv`.

### 2. Activate the Virtual Environment

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating the virtual environment is active.

### 3. Install Requirements

With the virtual environment activated, install all required dependencies:

```bash
pip install -r requirements.txt
```

This will install all packages listed in the `requirements.txt` file.

### 4. Navigate to LangGraph_Agentic_Deployment Directory

Navigate to the application directory:

```bash
cd LangGraph_Agentic_Deployment
```

### 5. Run the Application

Execute the Streamlit application:

```bash
streamlit run app.py
```
### 6. Configure API Keys

After running the application, set up your API key for either OpenAI or Anthropic.

**Note:** Make sure you use an API key from an account with a larger context window to run the application smoothly.

The application will start, and you can access it in your web browser at `http://localhost:8501` by default.

## Usage

Once the application is running in Streamlit:

1. Configure your preferred LLM provider (OpenAI or Anthropic)
2. Input your agentic workflow parameters
3. Execute and monitor the deployment process
4. View real-time logs and results in the Streamlit interface

## Troubleshooting

- **Module not found errors:** Ensure all requirements are installed and the virtual environment is activated
- **API key errors:** Verify your API key is correctly set in the environment variables
- **Port already in use:** Streamlit uses port 8501 by default. If it's in use, you can specify a different port:

```bash
streamlit run app.py --server.port 8502
```
## Requirements

All dependencies are listed in `requirements.txt`. Key packages include:

- LangGraph
- Streamlit
- OpenAI SDK (for OpenAI integration)
- Anthropic SDK (for Anthropic integration)

## Notes

- Ensure you have a valid API key from OpenAI or Anthropic with sufficient context window limits
- The application requires an active internet connection to communicate with the LLM APIs
- Keep your API keys secure and never commit them to version control

## Support

For issues or questions, please refer to the official documentation:

- [LangGraph Documentation](https://langchain.readthedocs.io/en/latest/modules/agents/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic Documentation](https://www.anthropic.com/)
