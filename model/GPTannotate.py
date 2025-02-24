import base64
import sys,os
from openai import OpenAI





try:
    tt = sys.argv[1]
except:
    tt = "sk-proj-j_6woUZh2M2khjgzVKwioXkZITNxxxxxxxxxxxxxxxxxxx"
print("Using api_key:{}".format(tt))
client = OpenAI(api_key=tt)
def encode_image(image_path):

    with open(image_path, "rb") as image_file:

        return base64.b64encode(image_file.read()).decode("utf-8")



# Path to your image
image_path = "./1.jpg"





# Getting the Base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a psychology expert, I need you to identify whether the subject in the video is stressed. Please find facial actions of the subject in this video.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },

            ],

        }

    ],

)
print(response.choices[0])