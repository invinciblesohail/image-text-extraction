# Image Text Extraction with Gemini 2.0 Flash

This repository contains a personal project demonstrating an image text extraction application powered by the Gemini 2.0 Flash large language model from Google.

## Overview

This application leverages the capabilities of the Gemini 2.0 Flash LLM (accessed via the Google Cloud AI API) to efficiently and accurately extract textual content from various image formats. It's built as a demonstration and exploration of the model's potential in optical character recognition (OCR) tasks.

## Features

* **Image to Text Conversion:** Extracts text from images provided as input.
* **Powered by Gemini 2.0 Flash:** Utilizes Google's fast and efficient large language model for text recognition.
* **Simple and User-Friendly Interface:** (If you have one, describe it briefly. E.g., "Built with Streamlit for an intuitive web interface.")


## How It Works

The application takes an image as input. This image is then processed using the Gemini 2.0 Flash LLM through API calls to Google Cloud AI. The model analyzes the visual content and returns the extracted textual information.

## Setup (for local development or running)

If you are interested in running this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/invinciblesohail/image-text-extraction.git
    cd your_repository_name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Google Cloud API credentials:**
    * You will need a Google Cloud Project with the Vertex AI API (or Google Cloud AI Platform API) enabled.
    * Create API credentials (an API key).
    * **Important:** For security, it is recommended to set the `GOOGLE_API_KEY` as an environment variable. You can do this by creating a `.env` file in the project root with the following content:
        ```
        GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"
        ```
        Make sure `.env` is in your `.gitignore` file.
    * Load the environment variable in your application code (see `main.py` for an example).

4.  **Run the application:**
    * (If using Streamlit)
        ```bash
        streamlit run main.py
        ```
    * (If using a different framework, provide the appropriate command)


## Potential Use Cases

* Extracting text from scanned documents.
* Reading text from images of signs or labels.
* Assisting with accessibility by converting image-based text.
* Personal experimentation with the Gemini 2.0 Flash LLM's vision capabilities.

## Notes and Limitations

* The accuracy of text extraction can depend on the quality and clarity of the input image.
* Performance may vary based on the complexity of the image and the length of the text.
* This is a personal project and may not be production-ready.
* Usage is subject to the terms of service of the Google Cloud AI API.

## Contributing

As this is a personal project, contributions are not actively sought at this time. However, feel free to fork the repository and experiment!



## Author

Sohail Ahemd/invinciblesohail
