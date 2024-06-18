import numpy as np
import pandas as pd
import re
import easyocr
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
import pymongo
import pandas as pd
import hydralit_components as hc
from streamlit_lottie import st_lottie
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import ast


# MongoDB connection string
mongo_atlas_user_name = st.secrets["mongo_atlas_user_name"]
mongo_atlas_password = st.secrets["mongo_atlas_password"]
# client = pymongo.MongoClient(f"mongodb+srv://{mongo_atlas_user_name}:{mongo_atlas_password}@cluster0.mkrsiyl.mongodb.net/?retryWrites=true&w=majority")
client = pymongo.MongoClient("mongodb://localhost:27017/")
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
            intro_card_extracting_animation_url = "https://lottie.host/e0b8a386-c9ea-4042-a9b8-ed63bec5f371/TF196S2MpR.json"
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

        st.write("Corrected Data:")
        st.dataframe(updated_df)
        


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
    


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])



def extract_text_and_display(image):
    # Perform OCR
    results = reader.readtext(image)
    text_lines = [text for bbox, text, _ in results]
    
    
    # Concatenate text lines into a single string
    text_to_classify = "\n".join(text_lines)
    def google_genai(text_to_classify):
        # Configure the Generative AI API
        genai.configure(api_key=st.secrets["google_genai_api_key"])
        # Define the generation configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Create the GenerativeModel with new configuration and safety settings
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        )

        # Start a chat session with the model
        chat_session = model.start_chat(history=[])

        # Send message to the model with extracted text
        response = chat_session.send_message(
            f"""Please analyze the following data and accurately identify the details. Organize them in the following format: 
            Company Name, Card Holder Name, Designation, Phone Number, Email, Website, Address, State, City, and Pincode. 
            Ensure the address is complete, including the city, state, and pincode. Please correct any errors in the given data like missing dot in the email or website ect. 
            Don't take company name from email id, if there is no company name in the provided data leave it blank.
            Provide the final output in a Python dictionary format.: {text_to_classify}""")
        
        return response
    
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
st.set_page_config(layout="wide")
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
import streamlit.components.v1 as components
import base64
from io import BytesIO

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
if menu_id == 'HOME':

    st.markdown("""
    <div style='text-align:center'>
        <img src='https://raw.githubusercontent.com/navinds/BizCardX-Extracting-Business-Card-Data-with-OCR/main/Media/bizcardx_how_it_works.svg' alt='Step 1: Load the Data' style='width: 100%; max-width: 1350px;'/>
    </div>
    """, unsafe_allow_html=True)   
    st.title("")
    st.subheader(":blue[Extract and Manage Business Card Information] ")
    hom1,hom2 = st.columns([1,1])
    with hom1:
            hom1_url = "https://lottie.host/934f12e6-2357-4a87-85d5-413a2023d82d/XXBSIeVZSq.json"
            hom1_ani = load_lottie_url(hom1_url)
            st_lottie(hom1_ani, width=350, height=350, quality='medium')        

    st.markdown("BizCardX uses state-of-the-art technology to organize your business card information seamlessly. Whether you're looking to organize contacts, save time, or improve efficiency, BizCardX is here to help.")


    # Features section
    st.subheader(":blue[Features]")
    st.markdown("""
    - **Accurate Information Extraction**: Efficiently extracts text information from business card images using EasyOCR.
    - **AI-Powered Identification**: Utilizes Google Gen AI (Gemini) to correctly identify and classify extracted data into relevant categories.
    - **User-Friendly Interface**: Seamlessly interact with a straightforward and user-friendly interface.
    - **Database Integration**: Efficiently store extracted information in MongoDB, an advanced NoSQL database, ensuring efficient data management and scalability.
    - **Manage Records**: Easily add, update, and delete records through the application interface.
    """)    
if menu_id == 'UPLOAD & EXTRACT':
    st.title("Business Card Data Extraction")
    st.caption("Simply Upload a business card, extract the data, and store into database! ")
    # Function to clear session state
    def clear_text_inputs():
        keys_to_delete = [key for key in list(st.session_state.keys()) if key.startswith('input_') or key in ['extracted_df', 'annotated_image']]
        for key in keys_to_delete:
            del st.session_state[key]

    # File uploader with on_change to clear text inputs
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], on_change=clear_text_inputs)
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
                                                style="width: 570px; height: 500px;" 
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
        st.markdown("Preview & Edit")

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

        # Show updated DataFrame
        st.dataframe(extracted_df, use_container_width=True)

        # Streamlit UI for the button
        if st.button('Save into Database'):
            mongo_extracted_data_table(extracted_df)

elif menu_id == 'VIEW OR MODIFY EXISTING DATA':
    modify_existing_data()