# wellness_checkin_chatbot

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Wellness_Checkin Chatbot is an innovative project aimed at creating a virtual caretaker using natural language processing (NLP) techniques. The model leverages a BERT encoder attached to an LSTM decoder to understand and generate conversations, helping users manage their day to day emotions effectively.

## Data Collection

The data used in this project has been collected from various sources, including:

- Kaggle datasets
- Hugging Face datasets
- Synthetic data generated using advanced language models

## Data Cleaning and Preprocessing

The raw data has been meticulously cleaned and preprocessed using the following libraries:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **nltk**: For natural language processing tasks such as tokenization, lemmatization, and stopwords removal
- **re**: For regular expressions to handle text cleaning

### Cleaning Process

The cleaning process involves:

1. Expanding contractions.
2. Converting text to lowercase.
3. Removing punctuation and numbers.
4. Tokenizing text.
5. Lemmatizing words.
6. Removing stopwords.

## Model Architecture

The model architecture consists of:

- **GPT-2 Decoder**: Finetune gpt2 model on huggingface transformers library with data <br/>

Other tested models include ->

- **BERT Encoder**: To encode the input text into meaningful embeddings.
- **LSTM Decoder**: To decode the embeddings and generate therapeutic responses.

This architecture allows the model to understand the context and nuances of the input text and provide coherent and contextually appropriate responses.

## Dependencies

To manage dependencies, this project uses Poetry. Ensure you have Poetry installed by following the instructions [here](https://python-poetry.org/docs/#installation).

Install the dependencies by running:

```bash
poetry install
```

## Usage

To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Humphery7/wellness_checkin_chatbot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd wellness_checkin_chatbot
   ```
3. Install all dependencies using Poetry:
   ```bash
   poetry install
   ```
4. Run the preprocessing script to clean your data.
5. Train the model using the cleaned data.

## Contributing

We welcome contributions to this project! If you have any suggestions, bug reports, or improvements, please open an issue or submit a pull request.

## License

This project is open source.

## Contact Information

**linkedin** : https://www.linkedin.com/in/humpheryotu/, **Email** : humpheryufuoma@gmail.com
