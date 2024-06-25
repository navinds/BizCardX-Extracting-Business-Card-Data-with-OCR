import numpy as np
import pandas as pd
import re
import easyocr
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pymongo
import pandas as pd
import hydralit_components as hc
from streamlit_lottie import st_lottie
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import ast
import streamlit.components.v1 as components
import base64
from io import BytesIO
from dotenv import load_dotenv
import os



st.set_page_config(page_title="BizCardX",layout="wide", page_icon="https://raw.githubusercontent.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/main/Media/bizcard_favicon.png",)

# MongoDB connection string
mongo_atlas_user_name = os.getenv("mongo_atlas_user_name")
mongo_atlas_password = os.getenv("mongo_atlas_password")
client = pymongo.MongoClient(f"mongodb+srv://{mongo_atlas_user_name}:{mongo_atlas_password}@cluster0.mkrsiyl.mongodb.net/?retryWrites=true&w=majority")
# client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client.bizcardx
collection = db.bizcard_collection

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
  

def mongo_extracted_data_table(extracted_df):
    for index, row in extracted_df.iterrows():
        mobile_number = row['mobile_number']
        
        # Check if the document already exists
        existing_doc = collection.find_one({'mobile_number': mobile_number})
        
        if existing_doc:
            # Update existing document
            collection.update_one(
                {'mobile_number': mobile_number},
                {'$set': {
                    'company_name': row['company_name'],
                    'card_holder_name': row['card_holder_name'],
                    'designation': row['designation'],
                    'email_address': row['email_address'],
                    'website_url': row['website_url'],
                    'address': row['address'],
                    'city': row['city'],
                    'state': row['state'],
                    'pin_code': row['pin_code']
                }}
            )
            st.success(f"The data already exists. Existing records have been updated.")
        else:
            # Insert new document
            collection.insert_one({
                'company_name': row['company_name'],
                'card_holder_name': row['card_holder_name'],
                'designation': row['designation'],
                'mobile_number': row['mobile_number'],
                'email_address': row['email_address'],
                'website_url': row['website_url'],
                'address': row['address'],
                'city': row['city'],
                'state': row['state'],
                'pin_code': row['pin_code']
            })
            st.success(f"Data has been successfully saved to the database.")
    
    print("Data updated successfully!")

# def enhance_image(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian Blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply adaptive thresholding
#     adaptive_threshold = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 11, 2)
    
#     # Apply dilation and erosion to connect broken text
#     kernel = np.ones((1, 1), np.uint8)
#     dilated = cv2.dilate(adaptive_threshold, kernel, iterations=1)
#     eroded = cv2.erode(dilated, kernel, iterations=1)
    
#     return eroded

# Function to fetch and display existing data
def fetch_existing_data():
    data = list(collection.find({}))
    return pd.DataFrame(data)

# Function to modify existing data
def modify_existing_data():
    st.title("Modify Existing Data")

    # Fetch existing data
    data_df = fetch_existing_data()
    
    if data_df.empty:
        st.write("No data found in the database.")
        return
    
    card_holder_list = data_df.apply(lambda row: f"{row['card_holder_name']} - {row['company_name']}", axis=1).tolist()
    selected_value = st.selectbox("Select Card Holder:", [''] + card_holder_list, key='selected_value',)
    if not selected_value:
        introcol1, introcol2, introcol3 = st.columns([1, 1, 1])
        with introcol2:
            intro_card_extracting_animation_url = "https://lottie.host/20ede9b0-bc49-4829-8387-75d2377c16e0/2Quqi5SjMT.json"
            intro_card_extracting_animation = load_lottie_url(intro_card_extracting_animation_url)
            st_lottie(intro_card_extracting_animation, width=350, height=350, quality='medium')       
    if selected_value:
        selected_card_holder_name = selected_value.split(' - ')[0]
        selected_record = data_df[data_df['card_holder_name'] == selected_card_holder_name].iloc[0]
    else:
        selected_record = None

    # Clear session state when a new card holder is selected
    if 'last_selected_value' not in st.session_state:
        st.session_state['last_selected_value'] = None

    if st.session_state['last_selected_value'] != selected_value:
        st.session_state['last_selected_value'] = selected_value
        for key in list(st.session_state.keys()):
            if key.startswith('input_'):
                del st.session_state[key]

    if selected_value:
        coll1, cooll2 = st.columns([1,0.15])
        with cooll2:
            if st.button("Delete Record"):
                # Delete the record from the MongoDB collection
                collection.delete_one({'_id': selected_record['_id']})
                st.success("Record has been successfully deleted.")
                st.experimental_rerun()
        with coll1:        
            st.subheader("Modify the selected record:")
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("Company Name", value=st.session_state.get('input_company_name', selected_record['company_name']), key='input_company_name')
            card_holder_name = st.text_input("Card Holder Name", value=st.session_state.get('input_card_holder_name', selected_record['card_holder_name']), key='input_card_holder_name')
            designation = st.text_input("Designation", value=st.session_state.get('input_designation', selected_record['designation']), key='input_designation')
            mobile_number = st.text_input("Mobile Number", value=st.session_state.get('input_mobile_number', selected_record['mobile_number']), key='input_mobile_number')
            email_address = st.text_input("Email Address", value=st.session_state.get('input_email_address', selected_record['email_address']), key='input_email_address')
        
        with col2:
            website_url = st.text_input("Website URL", value=st.session_state.get('input_website_url', selected_record['website_url']), key='input_website_url')
            address = st.text_input("Address", value=st.session_state.get('input_address', selected_record['address']), key='input_address')
            city = st.text_input("City", value=st.session_state.get('input_city', selected_record['city']), key='input_city')
            state = st.text_input("State", value=st.session_state.get('input_state', selected_record['state']), key='input_state')
            pin_code = st.text_input("Pin Code", value=st.session_state.get('input_pin_code', selected_record['pin_code']), key='input_pin_code')
        
        # Updated DataFrame with the edited values
        updated_df = pd.DataFrame({
            'company_name': [company_name],
            'card_holder_name': [card_holder_name],
            'designation': [designation],
            'mobile_number': [mobile_number],
            'email_address': [email_address],
            'website_url': [website_url],
            'address': [address],
            'city': [city],
            'state': [state],
            'pin_code': [pin_code]
        })

        st.subheader(":blue[Corrected Data:]")
        st.caption(":red[To see corrected data please click enter after making correction in the text box]")
        st.table(updated_df)
        


        if st.button("Update Record"):
            # Update the record in the MongoDB collection
            collection.update_one(
                {'_id': selected_record['_id']},
                {'$set': {
                    'company_name': company_name,
                    'card_holder_name': card_holder_name,
                    'designation': designation,
                    'mobile_number': mobile_number,
                    'email_address': email_address,
                    'website_url': website_url,
                    'address': address,
                    'city': city,
                    'state': state,
                    'pin_code': pin_code
                }}
            )
            st.success("Record has been successfully updated.")
            
            # Define the CSS style
            style = """
            <style>
                .info {
                    font-size: 15px;
                    line-height: 1.6;
                }
                .info .label {
                    color: white;
                }
                .info .value {
                    color: green;
                }
            </style>
            """

            # Write the style to the Streamlit app
            st.markdown(style, unsafe_allow_html=True)

            # Display each piece of information
            st.markdown(f"""
            <div class="info">
                <span class="label">company_name: </span><span class="value">{company_name}</span><br>
                <span class="label">card_holder_name: </span><span class="value">{card_holder_name}</span><br>
                <span class="label">designation: </span><span class="value">{designation}</span><br>
                <span class="label">mobile_number: </span><span class="value">{mobile_number}</span><br>
                <span class="label">email_address: </span><span class="value">{email_address}</span><br>
                <span class="label">website_url: </span><span class="value">{website_url}</span><br>
                <span class="label">address: </span><span class="value">{address}</span><br>
                <span class="label">city: </span><span class="value">{city}</span><br>
                <span class="label">state: </span><span class="value">{state}</span><br>
                <span class="label">pin_code: </span><span class="value">{pin_code}</span><br>
            </div>
            """, unsafe_allow_html=True)         
                            


# Initialize EasyOCR reader
@st.cache_resource
def easyocr_reader():
    reader = easyocr.Reader(['en'])
    return reader

reader = easyocr_reader()

@st.cache_resource
def load_generative_model():
    genai.configure(api_key=os.getenv("google_genai_api_key"))
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

generative_model = load_generative_model()

def google_genai(text_to_classify):
    # Start a chat session with the model
    chat_session = generative_model.start_chat(history=[])
    # Send message to the model with extracted text
    response = chat_session.send_message(
        f"""Please analyze the following data and accurately identify the details. Organize them in the following format: 
        Company Name, Card Holder Name, Designation, Phone Number, Email, Website, Address, State, City, and Pincode. 
        Ensure the address is complete, including the city, state, and pincode. Please correct any errors in the given data like missing dot in the email or website ect. 
        Don't take company name from email id, if there is no company name in the provided data leave it blank.
        Provide the final output in a Python dictionary format.: {text_to_classify}"""
    )
    return response

def extract_text_and_display(image):
    # Perform OCR
    results = reader.readtext(image)
    text_lines = [text for bbox, text, _ in results]
    
    
    # Concatenate text lines into a single string
    text_to_classify = "\n".join(text_lines)
    
    response = google_genai(text_to_classify)
    # Initialize extracted_info to avoid UnboundLocalError
    extracted_info = {
        'company_name': '',
        'card_holder_name': '',
        'designation': '',
        'mobile_number': '',
        'email_address': '',
        'website_url': '',
        'address': '',
        'city': '',
        'state': '',
        'pin_code': ''
    }

    # Parse the response based on the provided Python dictionary format
    if 'data =' in response.text:
        try:
            # Extract the dictionary content from the response text
            start_index = response.text.index('{')
            end_index = response.text.rindex('}') + 1
            dict_content = response.text[start_index:end_index]

            # Use ast.literal_eval to safely evaluate the dictionary string
            data = ast.literal_eval(dict_content)

            # Update extracted_info with the data from the response dictionary
            extracted_info.update({
                'company_name': data.get("Company Name", "").title() if data.get("Company Name") is not None else "",
                'card_holder_name': data.get("Card Holder Name", "").title() if data.get("Card Holder Name") is not None else "",
                'designation': data.get("Designation", "").title() if data.get("Designation") is not None else "",
                'mobile_number': data.get("Phone Number", ""),
                'email_address': data.get("Email", "").lower() if data.get("Email") is not None else "",
                'website_url': data.get("Website", "").lower() if data.get("Website") is not None else "",
                'address': data.get("Address", ""),
                'city': data.get("City", "").title() if data.get("City") is not None else "",
                'state': data.get("State", "").title() if data.get("State") is not None else "",
                'pin_code': data.get("Pincode", "")
            })
        except (ValueError, SyntaxError) as e:
            print("Error parsing response:", e)
    
    # Convert extracted_info dictionary to Pandas DataFrame
    extracted_df = pd.DataFrame([extracted_info])
    # Create annotated image with bounding boxes
    annotated_image = image.copy()
    for (bbox, text, prob) in results:
        # Extract bounding box coordinates
        (top_left, _, bottom_right, _) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw bounding box on the image
        cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

        # Calculate text position
        text_width, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_position = (bottom_right[0] + 10, top_left[1] + 25)
        if text_position[0] + text_width > image.shape[1]:
            text_position = (bottom_right[0] - text_width - 10, top_left[1] + 25)

        # Put OCR text and probability on the image
        cv2.putText(annotated_image, f'{text} ({prob:.2f})', text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Add marking line
        cv2.line(annotated_image, (text_position[0] - 5, top_left[1]), (text_position[0] - 5, bottom_right[1]), (0, 0, 255), 2)

    return extracted_df, annotated_image


# Streamlit UI
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2.2rem;
                    padding-left: 3.5rem;
                    padding-right: 3.5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:left'>
    <img src='https://raw.githubusercontent.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/main/Media/Bizcardx_page_logo.svg' style='width: 200px;'/>
</div>
""", unsafe_allow_html=True)



menu_data = [
    {'icon': "fa fa-home", 'label': "HOME"},
    {'icon': "fas fa-upload", 'label': "UPLOAD & EXTRACT"},
    {'icon': "far fa-edit", 'label': "VIEW OR MODIFY EXISTING DATA"},
    {'icon': "fa fa-info-circle", 'label': "ABOUT"}]

over_theme = {'txc_inactive': '#FFFFFF', 'bg': '#07bff'}  
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=False, 
    sticky_nav=True,
    sticky_mode='pinned')


def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
if menu_id == 'HOME':
    def home():
        st.markdown("""
        <div style='text-align:center'>
            <img src='https://raw.githubusercontent.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/main/Media/bizcardx_how_it_works_.svg' alt='Step 1: Load the Data' style='width: 100%; max-width: 1350px;'/>
        </div>
        """, unsafe_allow_html=True)   
        st.title("")
        st.subheader(":blue[Extract and Manage Business Card Information using our BizCardX] ")
        hom1,hom2 = st.columns(2)
        with hom1:
                hom1_url = "https://lottie.host/3634ced5-d95e-4064-bf0b-cda08e0b03f8/tgZ19X8CFL.json"
                hom1_ani = load_lottie_url(hom1_url)
                st_lottie(hom1_ani, width=500, height=400, quality='medium')        
        with hom2:
            st.text("")
            st.title("")
            st.subheader(":blue[BizCardX] uses state-of-the-art technology to organize your business card information seamlessly. Whether you're looking to organize contacts, save time, or improve efficiency, BizCardX is here to help.")


        # Features section
        
        hom3,hom4 = st.columns(2)
        with hom3:
            st.header(":blue[Features]")
            st.text("")
            st.text("")
            st.subheader(":blue[**Accurate Information Extraction**]: Efficiently extracts text information from business card images using EasyOCR.")

        with hom4:
                hom4_url = "https://lottie.host/6dafd6d6-6336-4d96-b325-c1dd97216ae4/NtQrq7nVIf.json"
                hom4_ani = load_lottie_url(hom4_url)
                st_lottie(hom4_ani,width=500, height=300, quality='medium')  

        hom5,hom6,hom7 = st.columns([1.15,1,1.90])
        with hom5:
            image_url = 'https://github.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/blob/main/Media/Google%20Ai%20Gemini.png?raw=true'
            st.image(image_url, width=400)  # Adjust width as needed
        with hom7:
            st.title(" ")
            st.subheader(":blue[**AI-Powered Identification**:] Utilizes Google Gen AI (Gemini) to correctly identify and classify extracted data into relevant categories.")
        with hom6:
            ai_url = "https://lottie.host/9f5e4af3-920f-4900-919a-1d0cb13a2dbf/3nfdf7Eoxz.json"
            ai_ani = load_lottie_url(ai_url)
            st_lottie(ai_ani, width=300, height=300, quality='medium')  


        hom7,hom8 = st.columns(2)
        with hom7:
            st.text("")
            st.title("")
            st.subheader(":blue[**User-Friendly Interface**:] Seamlessly interact with a straightforward and user-friendly interface. Enjoy quick access to essential features, intuitive navigation, and responsive design")

        with hom8:
                ui_url = "https://lottie.host/597857c7-1fff-423c-b104-f80bff449ed8/tKx86mrKLL.json"
                ui_ani = load_lottie_url(ui_url)
                st_lottie(ui_ani,width=500, height=350, quality='medium')  

        hom13,hom14,hom14a = st.columns([1.15,1.10,1.90])         
        
        with hom13:
            azure_image_url = 'https://raw.githubusercontent.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/main/Media/azure_logo.png'
            st.image(azure_image_url, width=350)  # Adjust width as needed
        with hom14a:
            st.title("")
            st.subheader(":blue[**Powered by Azure**:] Deployed on Microsoft Azure to ensure high performance, scalability, and to ensuring reliable operation with EasyOCR, OpenCV, and Google Gen AI (Gemini).")
        with hom14:
            speed_url = "https://lottie.host/11bed660-d9e6-4033-9220-a578b6208122/uVbvg9JuLM.json"
            speed_ani = load_lottie_url(speed_url)
            st_lottie(speed_ani, width=200, height=300, quality='medium') 

        hom9,hom10= st.columns(2)

        with hom9:
            st.title(" ")
            st.text("")
            st.text("")
            st.subheader(":blue[**Database Integration**:] Efficiently store extracted information in MongoDB, an advanced NoSQL database, ensuring efficient data management and scalability.")

        with hom10:
                db_url = "https://lottie.host/4a3c23db-b109-4ffa-976e-82b40ab9eb60/uuQ7e0SReD.json"
                db_ani = load_lottie_url(db_url)
                st_lottie(db_ani,width=500, height=400, quality='medium')  


        hom11,hom12 = st.columns(2)

        with hom11:
                ui_url = "https://lottie.host/20ede9b0-bc49-4829-8387-75d2377c16e0/2Quqi5SjMT.json"
                ui_ani = load_lottie_url(ui_url)
                st_lottie(ui_ani,width=500, height=350, quality='medium')  

        with hom12:
            st.title("")
            st.title("")
            st.subheader(":blue[**Manage Records**:] Easily add, update, and delete records through the application interface.")               

           
    home()

if menu_id == 'ABOUT':
    def about():
        st.title(":blue[About BizCardX]")

        st.markdown("""
        BizCardX is an advanced solution designed to streamline the process of managing business card information. Our application leverages cutting-edge technologies to ensure accurate and efficient extraction, identification, and management of business card data.
        """)
        
        st.header(":blue[Our Mission]")
        st.markdown("""
    Mission is to simplify the the way professionals handle business card information. We aim to provide a seamless and intuitive platform that saves time, reduces manual errors, and enhances productivity.
        """)
        
        
        st.header(":blue[Key Technologies]")
        st.markdown("""
        - **EasyOCR**: Utilized for high-accuracy text extraction from business card images.
        - **Google Gen AI (Gemini)**: Employed for intelligent data classification to correctly identify and categorize extracted text.
        - **MongoDB**: An advanced NoSQL database used for efficient and scalable data storage.
        - **Streamlit**: The framework used to build our user-friendly and interactive interface.
        - **Azure**: Cloud platform where the application is deployed for hosting and running the application.
        """)
        
        st.header(":blue[Core Features]")
        st.markdown("""
        - **Accurate Information Extraction**: Efficiently extracts text information from business card images.
        - **AI-Powered Identification**: Utilizes advanced AI to classify and organize extracted data into relevant categories.
        - **User-Friendly Interface**: Intuitive and responsive design for easy navigation and usage.
        - **Advanced Database Integration**: Stores extracted information in MongoDB for robust and scalable data management.
        - **Comprehensive Record Management**: Easily add, update, and delete records through the application interface.
        - **Microsoft Azure Deployment**: Deployed on Azure for optimal performance and to overcome memory limitations inherent in using EasyOCR, OpenCV, and Gemini.             
        """)
        
        st.header(":blue[Deployment]")
        st.markdown("""This application is deployed on Microsoft Azure to overcome memory limitations encountered with Streamlit's default hosting options. 
                    Azure provides robust cloud infrastructure that ensures reliable performance and scalability, allowing the application to handle large datasets and complex computations seamlessly.""")
        
        st.header(":blue[How This Application Will Be Helpful]")
        st.markdown("""
        1. Automates the process of extracting and organizing business card information, reducing manual data entry time.
        2. Minimizes human errors in data entry by accurately extracting and classifying information.
        3. Keeps all your business card information in one place, making it easy to manage and access contacts.
        4. Quickly retrieve contact details to follow up with leads, clients, and colleagues.
        5. Efficiently handles a growing number of business cards, making it suitable for individuals and businesses alike.
        """)
        

        st.header(":blue[About Me]")    
        st.markdown("""
            Hi, I'm Navin, deeply passionate about the sea of data science and AI. 
            My goal is to become a skilled data scientist.

            Beyond the lines of code, my aim is to innovate and be a part of transformative technological evolution. 
            The world needs solutions that not only solve problems but redefine them. 
            I'm here to create change.
        """)

        # LinkedIn link with logo
        st.header(":blue[Connect with Me or Share me your feedback ]")    
        col1, col2 = st.columns([1,20])
            
        with col1:  
            
            linkedin_logo = "https://img.icons8.com/fluent/48/000000/linkedin.png"  
            linkedin_url = "https://www.linkedin.com/in/navinkumarsofficial/"  
            st.markdown(f"[![LinkedIn]({linkedin_logo})]({linkedin_url})")
        with col2:
            # Email with logo
            email_logo = "https://img.icons8.com/fluent/48/000000/email.png"  
            your_email = "https://mail.google.com/mail/?view=cm&source=mailto&to=navin.workwise@gmail.com"
            st.markdown(f"[![Email]({email_logo})]({your_email})")
            
        st.header(":blue[My other projects]") 
        st.caption("Click the image to see the project")
        col1, col2 = st.columns([1, 5])
        with col1:  
            st.text("")
            fine_dine = "https://raw.githubusercontent.com/navinds/Zomato-Data-Analysis-and-Visualization/main/Media/finefdine_red_logo.png"  
            fine_dine_project_url = "https://navinsfinedine.streamlit.app/"  
            st.markdown(f'<a href="{fine_dine_project_url}" target="_blank"><img src="{fine_dine}" width="200"></a>', unsafe_allow_html=True)

        with col2:
            pulse_vision = "https://raw.githubusercontent.com/navinds/PhonePe-Pulse-Data-Visualization-and-Exploration/main/Media/pulse_vision_logo.png"  
            pulse_vision_project_url = "https://navinspulsevision.streamlit.app"
            st.markdown(f'<a href="{pulse_vision_project_url}" target="_blank"><img src="{pulse_vision}" width="200"></a>', unsafe_allow_html=True)

        st.header(":blue[Frequently Asked Questions]")

        with st.expander("How does BizCardX extract information from business cards?"):
            st.write("BizCardX uses EasyOCR to extract text from business card images and Google Gen AI (Gemini) to classify the extracted text into relevant categories.")

        with st.expander("Is my data secure with BizCardX?"):
            st.write("Yes, BizCardX ensures that your data is securely stored in MongoDB, an advanced NoSQL database known for its scalability and security features.")

        with st.expander("Can I update or delete saved records?"):
            st.write("Absolutely! BizCardX provides a user-friendly interface to easily add, update, and delete records.")

        with st.expander("What types of business cards does BizCardX support?"):
            st.write("BizCardX supports a wide variety of business cards, regardless of their layout or design, thanks to the robust OCR and AI technologies employed.")

        with st.expander("How can I provide feedback or request new features?"):
            st.write("We value your feedback! Please contact our support team or visit our website to provide feedback or request new features.")

        with st.expander("Can BizCardX handle business cards in multiple languages?"):
            st.write("Yes, BizCardX is equipped to handle business cards in multiple languages, leveraging EasyOCR's language support capabilities.")

        with st.expander("Do I need an internet connection to use BizCardX?"):
            st.write("Yes, an internet connection is required to upload images, process them using EasyOCR, and interact with Google Gen AI (Gemini) for data classification.")

        with st.expander("Is there a limit to the number of business cards I can upload?"):
            st.write("BizCardX does not impose a strict limit on the number of business cards you can upload. However, storage limitations may depend on the plan or subscription you are using.")

        with st.expander("How can I ensure the best OCR results with BizCardX?"):
            st.write("To ensure the best OCR results, please upload clear and well-lit images of the business cards. Avoid shadows and reflections that may obscure the text.")

        with st.expander("What happens if the OCR fails to extract information correctly?"):
            st.write("If the OCR fails to extract information correctly, you can manually edit the extracted data before saving it to the database.")

        with st.expander("Is there a mobile app for BizCardX?"):
            st.write("Currently, BizCardX is a web-based application. However, we are planning to develop a mobile app for on-the-go card scanning in the future.")

        with st.expander("Which programming language is used to build BizCardX?"):
            st.write("BizCardX is built using Python, leveraging libraries and frameworks such as Streamlit for the web interface, EasyOCR for optical character recognition, and various other tools for AI and database integration.")

        with st.expander("How long does it take to process a business card?"):
            st.write("Processing time can vary depending on the complexity of the card and server load, but typically it takes just a few seconds to extract and classify the information.")    

        with st.expander("Can I export the extracted data to other formats?"):
            st.write("Currently, BizCardX allows you to export extracted data in CSV format.")
    about()


if menu_id == 'UPLOAD & EXTRACT':
    st.title("Business Card Data Extraction")
    st.caption(":blue[Simply Upload a business card, extract the data, and store into database! ]")

    # Function to clear session state
    def clear_text_inputs():
        keys_to_delete = [key for key in list(st.session_state.keys()) if key.startswith('input_') or key in ['extracted_df', 'annotated_image']]
        for key in keys_to_delete:
            del st.session_state[key]

    # File uploader with on_change to clear text inputs
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], on_change=clear_text_inputs)
    st.markdown(
    """
    You can try using the following sample images for testing the application:  [Sample Images on Google Drive](https://drive.google.com/drive/folders/1ukKwUSMEYPzks8GdULB5dXZP6ZLDXvMe?usp=sharing)
    """
    )    
    if not uploaded_file:
        introcol1, introcol2, introcol3 = st.columns([1, 1, 1])
        with introcol2:
            intro_card_extracting_animation_url = "https://lottie.host/934f12e6-2357-4a87-85d5-413a2023d82d/XXBSIeVZSq.json"
            intro_card_extracting_animation = load_lottie_url(intro_card_extracting_animation_url)
            st_lottie(intro_card_extracting_animation, width=350, height=350, quality='medium')
    if uploaded_file is not None:
        # Read image from uploaded file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if 'extracted_df' not in st.session_state:
            extracting_animation_url = "https://lottie.host/3400bf08-d321-4dd1-9911-936065b9c4b9/6dwJipzxC4.json"
            extracting_animation = load_lottie_url(extracting_animation_url)
 
            animation_placeholder = st.empty()
            
            with st.spinner('Extracting text from image...'):
                if extracting_animation:
                    with animation_placeholder:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            uploaded_image = Image.open(uploaded_file)
                            uploaded_image_base64 = get_image_base64(uploaded_image)

                            # HTML and CSS to overlay the animation on the image
                            custom_html = f"""
                            <div style="position: relative; display: inline-block;">
                                <img src="data:image/png;base64,{uploaded_image_base64}" 
                                    style="width: 100%; height: auto;">
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                                    <lottie-player src="{extracting_animation_url}" 
                                                background="transparent" 
                                                speed="1" 
                                                style="width: 595px; height: 500px;" 
                                                loop autoplay>
                                    </lottie-player>
                                </div>
                            </div>
                            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest"></script>
                            """
                            components.html(custom_html, height=500)                            
                
                # Extract text and get results
                extracted_df, annotated_image = extract_text_and_display(image)
                st.session_state['extracted_df'] = extracted_df
                st.session_state['annotated_image'] = annotated_image

            # Clear the animation after extraction is complete
            animation_placeholder.empty()
        else:
            extracted_df = st.session_state['extracted_df']
            annotated_image = st.session_state['annotated_image']

        img_col1, img_col2 = st.columns(2, gap="large")
        
        # Display image
        with img_col1:
            st.image(image1, caption="Uploaded Image")

        with img_col2:
            st.image(annotated_image, caption="Annotated Image")

        # Display extracted information and input fields for correction
        st.subheader(":blue[Preview & Edit] :pencil:")
        st.caption("You can click the text box to edit data if any corrections is needed.")

        col1, col2 = st.columns(2)

        with col1:
            company_name = st.text_input("Company Name", value=st.session_state.get('input_company_name', extracted_df['company_name'].iloc[0]), key='input_company_name')
            card_holder_name = st.text_input("Card Holder Name", value=st.session_state.get('input_card_holder_name', extracted_df['card_holder_name'].iloc[0]), key='input_card_holder_name')
            designation = st.text_input("Designation", value=st.session_state.get('input_designation', extracted_df['designation'].iloc[0]), key='input_designation')
            mobile_number = st.text_input("Mobile Number", value=st.session_state.get('input_mobile_number', extracted_df['mobile_number'].iloc[0]), key='input_mobile_number')
            email_address = st.text_input("Email Address", value=st.session_state.get('input_email_address', extracted_df['email_address'].iloc[0]), key='input_email_address')
        with col2:
            website_url = st.text_input("Website URL", value=st.session_state.get('input_website_url', extracted_df['website_url'].iloc[0]), key='input_website_url')
            address = st.text_input("Address", value=st.session_state.get('input_address', extracted_df['address'].iloc[0]), key='input_address')
            city = st.text_input("City", value=st.session_state.get('input_city', extracted_df['city'].iloc[0]), key='input_city')
            state = st.text_input("State", value=st.session_state.get('input_state', extracted_df['state'].iloc[0]), key='input_state')
            pin_code = st.text_input("Pin Code", value=st.session_state.get('input_pin_code', extracted_df['pin_code'].iloc[0]), key='input_pin_code')

        # Update the extracted DataFrame with the corrected values
        extracted_df['company_name'].iloc[0] = company_name
        extracted_df['card_holder_name'].iloc[0] = card_holder_name
        extracted_df['designation'].iloc[0] = designation
        extracted_df['mobile_number'].iloc[0] = mobile_number
        extracted_df['email_address'].iloc[0] = email_address
        extracted_df['website_url'].iloc[0] = website_url
        extracted_df['address'].iloc[0] = address
        extracted_df['city'].iloc[0] = city
        extracted_df['state'].iloc[0] = state
        extracted_df['pin_code'].iloc[0] = pin_code

        ## Show updated DataFrame
        st.subheader(":blue[Extracted Data:]")
        st.caption(":red[If you did any correction, please click enter after making correction in the text box to update the correction]")
        st.table(extracted_df)
        
        st.caption(":blue[If everything is fine save the data into database. You can view or modify the data anytime.]")

        # Streamlit UI for the button
        if st.button('Save into Database'):
            mongo_extracted_data_table(extracted_df)

elif menu_id == 'VIEW OR MODIFY EXISTING DATA':
    modify_existing_data()
