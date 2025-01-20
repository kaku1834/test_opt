import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

def load_auth_config(yaml_path):
    """Load authentication configuration from YAML file"""
    with open(yaml_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config

def setup_authenticator(config):
    """Setup authentication with the provided configuration"""
    return stauth.Authenticate(
        credentials=config['credentials'],
        cookie_name=config['cookie']['name'],
        cookie_key=config['cookie']['key'],
        cookie_expiry_days=config['cookie']['expiry_days'],
    )

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Dashboard",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={}
    )

def handle_authentication(authenticator):
    """Handle the authentication process and return authentication status"""
    with st.container():
        authenticator.login()
    
    if st.session_state["authentication_status"]:
        st.sidebar.markdown(f'## Welcome *{st.session_state["name"]}*')
        authenticator.logout('Logout', 'sidebar')
        return True
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        return False
    else:
        st.warning('Please enter your username and password')
        return None

def initialize_authentication(yaml_path="config.yaml"):
    """Initialize authentication setup and handle the process"""
    config = load_auth_config(yaml_path)
    authenticator = setup_authenticator(config)
    return authenticator, handle_authentication(authenticator) 