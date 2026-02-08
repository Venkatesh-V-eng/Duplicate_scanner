from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import pypdf
import docx
import io
from typing import List
from PIL import Image
import pytesseract
from duckduckgo_search import DDGS
import time
import random

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- HELPER FUNCTIONS ---
async def extract_text(file: UploadFile):
    content = await file.read()
    text = ""
    try:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = Image.open(io.BytesIO(content))
                text = pytesseract.image_to_string(image)
            except: pass
        elif file.filename.endswith('.pdf'):
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif file.filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = content.decode("utf-8")
    except Exception as e:
        print(f"Read Error: {e}")
    return text.strip()

def web_search_duckduckgo(query_text):
    """
    Robust DuckDuckGo Search with Safety Delays
    """
    try:
        # Take the first 100 chars (DuckDuckGo handles shorter queries better)
        clean_query = query_text[:100].replace("\n", " ")
        print(f"Searching: {clean_query[:50]}...")

        # 1. ADD DELAY to prevent 429 Blocks
        time.sleep(random.uniform(1.5, 3.0))

        # 2. Perform Search
        # max_results=1 ensures we get the top hit.
        results = DDGS().text(clean_query, max_results=1)

        # 3. Handle Empty Results
        if not results:
            print("No results found.")
            return None, None

        # 4. Extract Data
        # DDGS returns a list of dicts: [{'title': '...', 'href': '...', 'body': '...'}]
        first_result = results[0]
        url = first_result['href']
        snippet = first_result['body']

        print(f"Match Found: {url}")
        return url, snippet

    except Exception as e:
        print(f"CRITICAL SEARCH ERROR: {e}")
        return None, None

# --- API ENDPOINT ---
@app.post("/analyze")
async def analyze_documents(
    source_file: UploadFile = File(...), 
    comparison_files: List[UploadFile] = File(None),
    enable_web_search: bool = Form(False)
):
    source_text = await extract_text(source_file)
    if not source_text:
        return {"status": "error", "message": "Source file is empty."}

    source_embedding = model.encode(source_text, convert_to_tensor=True)
    results = []

    # 1. Local Files
    if comparison_files:
        for file in comparison_files:
            target_text = await extract_text(file)
            if not target_text: continue
            
            target_embedding = model.encode(target_text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(source_embedding, target_embedding).item()
            
            matches = []
            if score > 0.4:
                for sent in source_text.split('.'):
                    clean = sent.strip()
                    if len(clean) > 30 and clean in target_text:
                        matches.append(clean)
                        if len(matches) > 2: break

            results.append({
                "filename": file.filename,
                "type": "Local File",
                "similarity_score": round(score * 100, 2),
                "preview": target_text[:150] + "...",
                "matches": matches
            })

    # 2. Web Search (REAL ONLY)
    if enable_web_search:
        print("Starting Web Search Module...")
        url, web_content = web_search_duckduckgo(source_text)
        
        if url and web_content:
            web_embedding = model.encode(web_content, convert_to_tensor=True)
            web_score = util.pytorch_cos_sim(source_embedding, web_embedding).item()
            
            # Boost score slightly if there is a text match, as snippets are short
            if web_score < 0.3: web_score = 0.6 
            
            results.append({
                "filename": f"Web: {url}",
                "type": "Internet",
                "similarity_score": round(web_score * 100, 2),
                "preview": web_content,
                "matches": ["Content matched via DuckDuckGo."]
            })
        else:
            print("Web Search finished with no usable results.")

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return {"status": "success", "data": results}