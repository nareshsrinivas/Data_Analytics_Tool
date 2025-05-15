import streamlit as st
from typing import Dict, Optional

# Store all the stock photo URLs
stock_photos = {
    "data analytics dashboard": [
        "https://pixabay.com/get/gf7c91b14cc6dd675b53803259bf2e2b5290e6d8a4c0f3c6b9765c595671d62fc3367dd571b3dcafb684ea77860f23c407e7a0dbf38002d91329249786edcecd5_1280.jpg",
        "https://pixabay.com/get/g860e4c73620fa83f66b99495e2b8b755aceafd84da7d146715f30076a4c5a65a2d09b4c9261815c89b969ca93aac517ecf55e419731281291777a3d0f3afa208_1280.jpg",
        "https://pixabay.com/get/gecf5bb35b37e3030cbe4fe71421bef1ebc59bb10f1f1c7620f8920cf9db0ffe373c6b19afd9ae827db118cbef5b0eb0c64ef6c8384de15b26e188441a0813a6b_1280.jpg",
        "https://pixabay.com/get/g7159b6f13546432b13cb894674860e644a89ebd3d1203a34ae389c78da0715b877f5ee5c074617686e12d32d037fd4f628d2ff9bb57b06dfa7e41ac5e434556e_1280.jpg"
    ],
    "data visualization charts": [
        "https://pixabay.com/get/g20b1185cce437ed18cbb2fcbdda65cac3635ef946e435cb763e87df141c7fbce5e73b67d1ffbdd72f927900a3fa9f366bd2edbcc5bda10b79ce016ab92dcb896_1280.jpg",
        "https://pixabay.com/get/g0983e84db13b9fd177c23c54a408741c79e895558265857fddea841d7cbecbea913b6a33d1125ea83ddb13d7faa8ce857d708ea207a0e92127529bb7de578f28_1280.jpg",
        "https://pixabay.com/get/gbc34523ffa3e9dd8c244f0deabe76c0ef69c1fa4df919e56447875bd38e5a2adc110d9ebfb44c3eb56d70a75d3782174980346ad8ed76c1c3b4c060111a2d31a_1280.jpg",
        "https://pixabay.com/get/g5bbdc3ca0da87238faea44d1260d6737e946368c8585f54a4dae64312cd40c421d0faf4448460bb82aaebcadfc9f7b4cb0fec0683882e03bde9c2ddc2daa486c_1280.jpg"
    ],
    "business analytics interface": [
        "https://pixabay.com/get/g6804f8315ec309e073d7f1b61f06d23807f5cb8d1918f95871cf8f91df5b57ca4944cb9ef741f3c20d2c21f7fc3e8ab7263ad7a73272a036ed0a8d4ae61ddf75_1280.jpg",
        "https://pixabay.com/get/g03119c39a5f143c48e8a4efa5906e36d711f8e6fe1423bd4a272b00a1d463ae7a621d45819604edda63fa2ca542afabc45f1057f783def4488804ee2d88cb497_1280.jpg"
    ]
}

def display_header_image(category: str, height: int = 300) -> None:
    """
    Display a header image from the specified category of stock photos
    
    Parameters:
    category - The category of stock photo to display
    height - The height of the image in pixels (default: 300)
    """
    # Check if the category exists
    if category not in stock_photos:
        st.warning(f"No stock photos available for category: {category}")
        return
    
    # Initialize session state for used photos if it doesn't exist
    if "used_photos" not in st.session_state:
        st.session_state["used_photos"] = {}
    
    # Initialize category in used_photos if not present
    if category not in st.session_state["used_photos"]:
        st.session_state["used_photos"][category] = []
    
    # Find an unused photo
    available_photos = [photo for photo in stock_photos[category] 
                        if photo not in st.session_state["used_photos"][category]]
    
    # If all photos have been used, reset the used photos for this category
    if not available_photos:
        st.session_state["used_photos"][category] = []
        available_photos = stock_photos[category]
    
    # Select the first available photo
    photo_url = available_photos[0]
    
    # Mark as used
    st.session_state["used_photos"][category].append(photo_url)
    
    # Display the image with centered alignment
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="{photo_url}" alt="{category}" style="max-height: {height}px; max-width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )

def get_photo_url(category: str, index: Optional[int] = None) -> str:
    """
    Get a URL for a stock photo from the specified category
    
    Parameters:
    category - The category of stock photo
    index - Optional specific index to use (default: None, which selects the next unused photo)
    
    Returns:
    URL of the selected stock photo
    """
    # Check if the category exists
    if category not in stock_photos:
        return ""
    
    # If index is specified, use that photo
    if index is not None and 0 <= index < len(stock_photos[category]):
        return stock_photos[category][index]
    
    # Initialize session state for used photos if it doesn't exist
    if "used_photos" not in st.session_state:
        st.session_state["used_photos"] = {}
    
    # Initialize category in used_photos if not present
    if category not in st.session_state["used_photos"]:
        st.session_state["used_photos"][category] = []
    
    # Find an unused photo
    available_photos = [photo for photo in stock_photos[category] 
                        if photo not in st.session_state["used_photos"][category]]
    
    # If all photos have been used, reset the used photos for this category
    if not available_photos:
        st.session_state["used_photos"][category] = []
        available_photos = stock_photos[category]
    
    # Select the first available photo
    photo_url = available_photos[0]
    
    # Mark as used
    st.session_state["used_photos"][category].append(photo_url)
    
    return photo_url
