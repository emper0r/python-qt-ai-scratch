# Educational AI - Neural Network Visualizer

This project is an educational tool designed to visualize the internal workings of a simple neural network in real-time. It demonstrates how a neural network learns patterns in text data, either at the character level or word level. Built with Python and PyQt6, it features a graphical interface that displays neuron activations, predictions, and the learning process.

## Features

- **Real-time Visualization**: Watch hidden neuron activations updates as the network learns.
- **Dual Modes**: Switch between Character-level (simpler) and Word-level (more complex) learning.
- **Configurable Architecture**: Adjust hidden neurons, learning rate, and input window size via `config.json`.
- **Interactive Training**: The AI learns from text streams and attempts to predict the next token.
- **Data Sources**: Can load text from local files (txt, pdf) or URLs.

## Screenshot
![demo](https://github.com/emper0r/python-qt-ai-scratch/blob/main/edu_ai_demo.png)


## Installation

To run this project, you need Python 3 installed on your system. It is recommended to use a virtual environment.

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repo cloned>
```

### 2. Set up a Virtual Environment

Create a virtual environment to isolate the dependencies:

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt (wait until finish)
```

## Usage

Run the main script to start the application:

```bash
python edu_ai.py
```

The application window will open, showing the neuron visualizer, predictions table, and training status. The AI will start learning based on the initial configuration.

## Configuration

You can customize the neural network parameters by editing the `config.json` file:

```json
{
    "hidden_neurons": 32,      // Number of neurons in the hidden layer
    "learning_rate": 0.1,      // Learning rate for backpropagation
    "input_window": 5,         // Number of previous tokens to consider as context
    "thinking_delay": 0.05,    // Delay in seconds between training steps (for visualization)
    "visualization_mode": "heatmap",
    "token_mode": "word"       // "char" or "word" mode
}
```

## Dependencies

- [NumPy](https://numpy.org/): For matrix operations and neural network math.
- [PyQt6](https://pypi.org/project/PyQt6/): For the graphical user interface.
- [Requests](https://docs.python-requests.org/): For fetching text from URLs.
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/): For parsing HTML content.
- [pypdf](https://pypi.org/project/pypdf/): For reading PDF files.

## License

This project is for educational purposes.
