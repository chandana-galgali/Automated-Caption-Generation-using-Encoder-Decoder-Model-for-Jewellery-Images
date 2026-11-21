# LLM baseline (Gemini), experimental

An early experiment that sends the image to Google Gemini (`gemini-2.5-flash`) and asks it for the jewellery type and a one-line description. It is **not** used by the main application, which runs the trained VGG-16 + GRU models in `backend/`.

```bash
pip install -r requirements.txt
# create a .env file containing: GEMINI_API_KEY=your_key   (never commit it)
uvicorn main:app --reload --port 8001
```
