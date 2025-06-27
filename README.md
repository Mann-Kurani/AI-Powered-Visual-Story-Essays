# 📚 AI-Powered Visual Story Essays

Transform classic literature into immersive visual essays using AI. This project combines **Google Gemini** for intelligent text summarization and **Stable Diffusion** for visual generation — all served through a lightweight **Flask** web app.

The web app uses **Hugging Face’s hosted `stable-diffusion-v1-5` model** for fast image generation during inference.  
Additionally, I tested and verified the image generation pipeline using a **custom-built Stable Diffusion model from scratch**, implemented locally in **PyTorch**.

It includes a fully custom-built PyTorch Stable Diffusion pipeline with encoder-decoder, CLIP embeddings, and DDPM sampling.

> 📂 Some components (`pipeline.py`, `model_loader.py`, `init.py`) are adapted from my own [Stable Diffusion from Scratch](https://github.com/Mann-Kurani/Stable-Diffusion-from-Scratch) repository.

---

## 🚀 Features

- 🔎 **Context-Aware Summarization**  
  Summarizes user-input passages from literature into coherent, character-based plot points using **Google Gemini API**.

- 🎨 **Image Generation via Stable Diffusion**  
  Each summarized point is visually illustrated using **Hugging Face's `stable-diffusion-v1-5`**, guided by high CFG scales for better prompt adherence.

- 🧠 **Custom Stable Diffusion Pipeline (from Scratch)**  
  A separate module rebuilds the Stable Diffusion process using **PyTorch**, including:
  - VAE encoder-decoder  
  - CLIP-based context embedding  
  - DDPM-based diffusion sampler  
  - Tokenization with Hugging Face's `CLIPTokenizer`

- 🖼️ **Image Rendering with Titles**  
  Each generated image is overlaid with its corresponding story point using `Pillow`.

- 🌐 **Flask Web App Interface**  
  A lightweight web UI for entering paragraph input, triggering image generation, and viewing the resulting visual story.

---

## 📂 Project Structure

```

├── app.py                              # Flask web application
├── pipeline.py                         # Custom Stable Diffusion pipeline
├── model\_loader.py                     # Loads models from .ckpt
├── init.py                             # Downloads tokenizer and model weights
├── stable\_diffusion\_model\_from\_scratch.ipynb  # Colab Notebook
├── static/
│   └── images/                         # Auto-generated image output
├── templates/
│   └── index.html                      # Web app frontend

````

---

## 📸 Sample Workflow

1. 📖 Input a paragraph from a literary text (e.g., *Frankenstein*).  
2. 🤖 Gemini LLM condenses it into 5 key plot points.  
3. 🧠 Each point is passed through the Stable Diffusion API (or custom PyTorch pipeline).  
4. 🖼️ Images are generated and overlaid with the corresponding text.  
5. 🔗 All visuals are displayed on the results page.

---

## 🔑 API Setup

Create an `auth.py` file with your API keys:

```python
def get_gemini_api_key():
    return "YOUR_GEMINI_API_KEY"

def get_hf_api_key():
    return "YOUR_HUGGINGFACE_API_KEY"
````

---

## 🧪 Try the PyTorch Pipeline

For full training/inference from scratch, run `stable_diffusion_model_from_scratch.ipynb` in Google Colab.

---

## 📘 Academic Context

This project was completed as part of my final academic coursework, exploring generative AI applications for creative storytelling. It merges foundational ML concepts with modern diffusion-based generation, offering a hands-on look at how LLMs and image models can collaborate creatively.

---

## 🧠 Learnings

* Prompt engineering for LLM-image synergy
* Working with Gemini and Hugging Face Inference APIs
* Building inference pipelines from scratch using PyTorch
* Rendering and annotating AI-generated images

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

* [Google Gemini](https://ai.google.dev/)
* [Hugging Face](https://huggingface.co/)
* [Project Gutenberg](https://www.gutenberg.org/) for source texts

