import os
import tempfile
import time
import fitz
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from google.cloud import vision
from google.cloud.vision_v1 import types

_vision_async_client = None
def get_vision_async_client():
    global _vision_async_client
    if _vision_async_client is None:
        _vision_async_client = vision.ImageAnnotatorClient()
    return _vision_async_client

def submit_pdf_to_vision_async(pdf_bytes):
    client = get_vision_async_client()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, 'rb') as f:
            content = f.read()
        input_config = types.InputConfig(mime_type='application/pdf', content=content)
        features = [types.Feature(type_=types.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        request = types.AsyncBatchAnnotateFilesRequest(
            requests=[types.AsyncAnnotateFileRequest(input_config=input_config, features=features)]
        )
        operation = client.async_batch_annotate_files(request=request)
        return operation.operation.name
    finally:
        os.unlink(tmp_path)

def poll_vision_operation(operation_name):
    client = get_vision_async_client()
    operation = client.transport.operations_client.get_operation(operation_name)
    if not operation.done:
        return None
    response = operation.response
    text_pages = []
    for file_response in response.responses:
        for page in file_response.responses:
            if page.full_text_annotation:
                text_pages.append(page.full_text_annotation.text)
    full_text = '\n\n'.join(text_pages)
    return full_text

def extract_pdf_with_tesseract(pdf_bytes):
    """Process PDF locally with Tesseract (lower DPI for speed)."""
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = 150 / 72   # 150 DPI – faster than 300
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Preprocess
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        text = pytesseract.image_to_string(img, config='--psm 6 --oem 3 -l eng')
        all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)