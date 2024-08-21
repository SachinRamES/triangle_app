import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to extract triangle sides from image
def extract_triangle_sides(image):
    # Convert image to OpenCV format
    img = np.array(image.convert('RGB'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the triangle
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Get the vertices
    vertices = [point[0] for point in approx]
    if len(vertices) != 3:
        st.error("The detected shape is not a triangle")
        return None
    
    # Calculate side lengths
    def distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    a = distance(vertices[0], vertices[1])
    b = distance(vertices[1], vertices[2])
    c = distance(vertices[2], vertices[0])
    
    return a, b, c

# Function to calculate trigonometric values
def calculate_trigonometry(a, b, c):
    def cos_angle(a, b, c):
        return (a**2 + b**2 - c**2) / (2 * a * b)
    
    A_rad = np.arccos(cos_angle(b, c, a))
    B_rad = np.arccos(cos_angle(a, c, b))
    C_rad = np.arccos(cos_angle(a, b, c))
    
    return {
        'A_deg': np.degrees(A_rad),
        'B_deg': np.degrees(B_rad),
        'C_deg': np.degrees(C_rad),
        'sin_A': np.sin(A_rad),
        'sin_B': np.sin(B_rad),
        'sin_C': np.sin(C_rad),
        'cos_A': np.cos(A_rad),
        'cos_B': np.cos(B_rad),
        'cos_C': np.cos(C_rad),
        'tan_A': np.tan(A_rad),
        'tan_B': np.tan(B_rad),
        'tan_C': np.tan(C_rad)
    }

# Streamlit app
st.title('Triangle Properties Calculator')

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    sides = extract_triangle_sides(image)
    
    if sides:
        a, b, c = sides
        st.write(f"Sides of the triangle: a={a:.2f}, b={b:.2f}, c={c:.2f}")
        
        results = calculate_trigonometry(a, b, c)
        st.write("Trigonometric Values:")
        st.write(f"Angle A: {results['A_deg']:.2f} degrees")
        st.write(f"Angle B: {results['B_deg']:.2f} degrees")
        st.write(f"Angle C: {results['C_deg']:.2f} degrees")
        st.write(f"sin(A): {results['sin_A']:.2f}")
        st.write(f"cos(A): {results['cos_A']:.2f}")
        st.write(f"tan(A): {results['tan_A']:.2f}")
        st.write(f"sin(B): {results['sin_B']:.2f}")
        st.write(f"cos(B): {results['cos_B']:.2f}")
        st.write(f"tan(B): {results['tan_B']:.2f}")
        st.write(f"sin(C): {results['sin_C']:.2f}")
        st.write(f"cos(C): {results['cos_C']:.2f}")
        st.write(f"tan(C): {results['tan_C']:.2f}")
