# For deployment
import streamlit as st

# For YOLOv8 model
import yolo

def start():
    # Default
    clicked = False

    # Main app
    st.markdown("# Group Ginger 🫚\n## Medicine Detection and Classification")
    st.write(f"The images uploaded were identified and classified by YOLOv8 model. Some detections might be incorrect.")
    
    # Upload an image
    st.markdown("### Test the model")
    data = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    clicked = st.button("Upload")

    # Predict image
    st.markdown("### Results")
    con = st.container()

    if clicked == True:
        try:
            with st.spinner("Loading results..."):
                image = yolo.get_predicted_image(data)
                con.image(image=image, channels="RGB")
                con.success("Success!")
        except:
            con.error("Error predicting image.")
            
    # Show result
    st.write("This is where the results will be shown.")

    # Member list
    st.markdown("""
    ## Ginger Members 🫚\n
    ### Split 1
    1. Ausan, Reizen Kim
    2. Layson, Sebastian Carlo Enrique Nicolas
    ### Split 2
    1. Mayordo, Zherish Galvin
    2. Villasan, John Michael
    ### Split 3
    1. Bayoneta, Allen Jethro
    2. Valeroso, Sarah Mae
    """)

    # Colab links ☁️
    st.markdown("""
    ## Colab Links
    1. [YOLOv8](https://colab.research.google.com/drive/118JLbYbrDQxgMO4bF7NkTGKhoVfIqq3i?usp=sharing)
    2. [EfficientNet](https://colab.research.google.com/drive/1pX-uae1fgC9JxiuVCLnw8iKnRHhb0h_S?usp=sharing)
    3. [CNN-MLP](https://colab.research.google.com/drive/10-C_6o2dxxH93m7DVTzmAXCXiUvrqrfC?usp=sharing)
""")


if __name__ == "__main__":
    start()
