# import streamlit as st

# #create a app to recommend food
# st.title('Food Recommendation App')

# user_prompt ,user_like , user_dislike = st.text_input("Where are you?"), st.text_input("What do you like?"), st.text_input("What do you dislike?")

# if st.button('Generate') and user_prompt and user_like and user_dislike:
#     with st.spinner('Generating.....'):
#         # prepare input variables
#         input_variables = {
#             'user_prompt': user_prompt,
#             'user_like': user_like,
#             'user_dislike': user_dislike
#         }
 
 
    
# Run this app by running the following command in your terminal:
import streamlit as st

# Create a food recommendation app
st.title('Food Place Recommendation App')

# Get user input
user_location = st.text_input("Where are you located?")
user_likes = st.text_input("What type of food do you like?")
user_dislikes = st.text_input("Are there any specific cuisines you dislike?")

# Generate recommendation on button click
if st.button('Find Best Food Places') and user_location and user_likes and user_dislikes:
    with st.spinner('Finding the best food places for you...'):
        # Prepare input variables
        input_data = {
            'user_location': user_location,
            'user_likes': user_likes,
            'user_dislikes': user_dislikes
        }

        # Add your recommendation logic here (not provided in the original code)
        # recommendation = generate_recommendation(input_data)

        # Display recommendation (replace this line with your actual recommendation)
        st.success(f"Here are the best food places near {user_location} based on your preferences: {recommendation}")
