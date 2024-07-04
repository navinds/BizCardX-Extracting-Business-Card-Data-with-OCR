# BizCardX: Extracting Business Card Data with OCR

BizCardX is a Streamlit application that allows users to upload images of business cards and extract relevant information using EasyOCR and Google Gen AI (Gemini). The application processes the images to identify and classify text into categories such as company name, card holder name, designation, phone number, email address, website URL, area, city, state, and pin code. The extracted information can be saved in a MongoDB database, and users can view, update, and delete records through the Streamlit interface.
[click here to see the app](https://navins-bizcardx.azurewebsites.net/)

![image](https://github.com/navinds/Zomato-Data-Analysis-and-Visualization/assets/155221787/d8d19423-c7c7-4099-a45d-d0ad84f65572)
![image](https://github.com/navinds/Zomato-Data-Analysis-and-Visualization/assets/155221787/507bb3a4-ee05-4dd4-a475-2e5b4d1463ad)
![image](https://github.com/navinds/Zomato-Data-Analysis-and-Visualization/assets/155221787/0f1db7e4-6a64-4cba-ae67-fee3011b99cb)
![image](https://github.com/navinds/Zomato-Data-Analysis-and-Visualization/assets/155221787/7c843122-54f2-4cc2-bef6-59c2056bbcf8)

## Table of Contents
1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Technologies Used](#technologies-used)
4. [How This Application Will Be Helpful](#how-this-application-will-be-helpful)
5. [Installation](#installation)
6. [References](#references)
7. [License](#license)
8. [About the Developer](#about-the-developer)
9. [Contact Information](#contact-information)

## Features
- **Accurate Information Extraction**: Efficiently extracts text information from business card images using EasyOCR.
- **AI-Powered Identification**: Utilizes Google Gen AI (Gemini) to correctly identify and classify extracted data into relevant categories.
- **User-Friendly Interface**: Easy-to-use interface with quick access to essential features and intuitive navigation.
- **Advanced Database Integration**: Save extracted information in MongoDB, ensuring efficient data management and scalability.
- **Manage Records**: Easily add, update, and delete records through the application interface.
- **Microsoft Azure Deployment**: Deployed on Azure for optimal performance and to overcome memory limitations inherent in using EasyOCR, OpenCV, and Gemini.
  
## How It Works
1. **Upload a Business Card Image**: Use the 'Upload Business Card' section to upload an image of a business card.
2. **Extract Information**: BizCardX uses EasyOCR to extract text from the image.
3. **AI-Powered Data Identification**: Google Gen AI (Gemini) identifies and classifies the extracted text into categories such as company name, card holder name, designation, phone number, email address, website URL, area, city, state, and pin code.
4. **Review and Save**: Review the extracted information and save it to your database for easy management.
5. **Manage Records**: View, update, or delete saved business card information from the 'View Records' section.

## Technologies Used
- **Python**: Programming language used for developing the application.
- **EasyOCR**: OCR engine for extracting text from images.
- **Google Gen AI (Gemini)**: AI service used for data classification to correctly identify and categorize extracted text.
- **MongoDB**: NoSQL database for storing extracted information.
- **OpenCV**: Used for handling images; includes preprocessing steps such as resizing, cropping, and thresholding to enhance image quality before passing it to the OCR engine.
- **Streamlit**: Framework for creating the web interface.
- **Streamlit Lottie**: For adding animations to the UI.
- **Azure**: Cloud platform where the application is deployed for hosting and running the application.

## Deployment
This application is deployed on Microsoft Azure to overcome memory limitations encountered with Streamlit's default hosting options. Azure provides robust cloud infrastructure that ensures reliable performance and scalability, allowing the application to handle large datasets and complex computations seamlessly.

## How This Application Will Be Helpful
1. Automates the process of extracting and organizing business card information, reducing manual data entry time.
2. Minimizes human errors in data entry by accurately extracting and classifying information.
3. Keeps all your business card information in one place, making it easy to manage and access contacts.
4. Quickly retrieve contact details to follow up with leads, clients, and colleagues.
5. Efficiently handles a growing number of business cards, making it suitable for individuals and businesses alike.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR.git
    cd BizCardX-Extracting-Business-Card-Data-with-OCR
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `config.toml` file**:
    - Inside the `.streamlit` folder (create the folder if it doesn't exist), create a `config.toml` file and add your passwords or other sensitive information in it. Here is an example structure:
        ```toml
        [general]
        mongo_atlas_user_name = "your_mongodb_username"
        mongo_atlas_password = "your_mongodb_password"
        google_genai_api_key= "your_gemini_api_key"
        ```

5. **Run the application**:
    ```bash
    streamlit run main.py
    ```

## Access the App

Explore the live version of the Fine Dine app here: [BizCardX App](https://navins-bizcardx.azurewebsites.net/).

## References
- [Python Documentation](https://docs.python.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [EasyOCR Documentation](https://www.jaided.ai/easyocr/documentation/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [MongoDB Documentation](https://www.mongodb.com/docs/)

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## About the Developer
BizCardX is developed by Navin Kumar S, a dedicated tech enthusiast with a passion for the sea of data science and AI. My goal is to become a skilled data scientist.

Beyond the lines of code, my aim is to innovate and be a part of transformative technological evolution. The world needs solutions that not only solve problems but redefine them. I'm here to create change.

## Contact Information
- **LinkedIn:** [Navin](https://www.linkedin.com/in/navinkumarsofficial/)
- **Email:** navin.workwise@gmail.com

Feel free to connect with me on LinkedIn or reach out via email for any inquiries or collaboration opportunities.
