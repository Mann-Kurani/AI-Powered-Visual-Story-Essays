from flask import Flask, render_template, request, jsonify
import os
import ast
from google import genai
from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt
from auth import get_gemini_api_key, get_hf_api_key
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Ensure 'static/images' exists for saving images
if not os.path.exists("static/images"):
    os.makedirs("static/images")

def wrap_text(text, font, max_width):
    """Wrap text so it fits within max_width."""
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and font.getlength(line + words[0]) <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines

@app.route('/')
def home():
    return render_template('index.html')

client = genai.Client(api_key=get_gemini_api_key())
hf_client = InferenceClient(provider="hf-inference", api_key=get_hf_api_key())

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get("paragraph", "")
    
    # Gemini API for summarization
    # content = f""" The narrator is Frankenstein FYI
    # I want to make a story essay with the given context story. Only give me 5 points that summarizes it from third person pronouns and nouns (Use character names only everytime and make their character descriptive everytime. Refrain from using vague words like stranger, guest, etc.), 
    # make sure the story has a good flow and give the output in the form of a python list.
    # context: '{user_input}' """
    content = f"""The narrator is Frankenstein FYI
    I want to make a story essay with the given context story. 
    Only give me 5 points that summarizes it from third person pronouns and nouns (Use character names), 
    make sure the story has a good flow and give the output in the form of a python list.
    context: '{user_input}'"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=content)
    # '```python\n[\n    "Frankenstein develops a deep affection for his guest, admiring his wisdom and eloquence while pitying his obvious misery.",\n    
    # "Frankenstein confides in the guest about his ambitious enterprise, expressing his willingness to sacrifice everything for its success, which elicits a strong, 
    # negative emotional reaction from the guest.",\n    "The guest, deeply affected by Frankenstein\'s words, warns him against pursuing his ambition, 
    # hinting at a shared \'madness\' and offering to reveal his own tale as a cautionary example.",\n    "The guest recounts his past desire for a true friend 
    # and expresses his belief that such a bond is essential for human fulfillment, lamenting his own loss and despair.",\n    
    # "Despite his grief, the guest finds solace in nature\'s beauty, showcasing a resilience and depth that further captivates Frankenstein, who sees him as an extraordinary 
    # individual with unparalleled insight and expressiveness."\n]\n```\n')]
    story_summary = ast.literal_eval(response.text.replace("\\n", "").replace("```", "").replace("python", "").strip())
    # print(response)
    # Stable Diffusion API for image generation
    uncond_prompt = "blurry, low quality, distorted"
    cfg_scale = 10 # min: 1, max: 14. Controls the strength of guidance toward the PROMPT
    # Low numbers (0-6 ish): you're telling the application to be free to ignore your prompt
    # Mid-range (6-10 ish): you're telling the application you'd like it to do what you're asking, but you don't mind a bit of variation
    # High numbers (10+): you're telling the application to do exactly what you tell it, with no deviation
    num_inference_steps = 100
    seed = None

    image_urls = []
    
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    for i, prompt in enumerate(story_summary, start=1):
        print(prompt)
        image = hf_client.text_to_image(
            prompt, 
            model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            negative_prompt=uncond_prompt, 
            guidance_scale=cfg_scale,
            num_inference_steps=num_inference_steps, 
            seed=seed
        )
        

        # # Display and save the image with title using matplotlib
        # plt.imshow(image)
        # plt.title(prompt, fontsize=12, wrap=True)  # Add title
        # plt.axis("off")
        # # plt.show()

        #     # Save the image inside the 'images' folder
        # image_path = f"static/images/img{i}.jpg"
        # plt.savefig(image_path, bbox_inches="tight", pad_inches=0.2)  # Save with padding
        # plt.close() 
        # image_urls.append(image_path)

        # Convert image to RGB format
        image = image.convert("RGB")


        # Add title to the image
        draw = ImageDraw.Draw(image)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust if needed
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)

        # Get image dimensions
        img_width, img_height = image.size
        max_text_width = img_width - 20  # Allow padding
        wrapped_lines = wrap_text(prompt, font, max_text_width)

        # Calculate total text height
        text_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in wrapped_lines) + (10 * len(wrapped_lines))

        # Draw background rectangle
        text_bg_height = text_height + 10
        draw.rectangle([(0, img_height - text_bg_height), (img_width, img_height)], fill="white")

        # Draw wrapped text
        y_text = img_height - text_bg_height + 5
        for line in wrapped_lines:
            text_x = (img_width - font.getlength(line)) // 2  # Center text
            draw.text((text_x, y_text), line, fill="black", font=font)
            y_text += font.getbbox(line)[3] - font.getbbox(line)[1] + 5

        # Save the image
        image_path = f"static/images/img{i}.jpg"
        image.save(image_path)
        image_urls.append(image_path)

    return jsonify({"images": image_urls})
    
if __name__ == '__main__':
    app.run(debug=True)
