import re
import os
import time
import json
from google.cloud import vision
from google.oauth2 import service_account
from google.cloud.vision_v1 import types
import tempfile

_vision_client = None
_vision_async_client = None

def get_vision_client():
    global _vision_client
    if _vision_client is None:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            credentials = service_account.Credentials.from_service_account_file(creds_path)
            _vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            _vision_client = vision.ImageAnnotatorClient()
    return _vision_client

def get_vision_async_client():
    global _vision_async_client
    if _vision_async_client is None:
        _vision_async_client = get_vision_client()  # same client works for async
    return _vision_async_client

def extract_with_google_vision(image_bytes):
    client = get_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'Vision API error: {response.error.message}')
    annotation = response.full_text_annotation
    text = annotation.text if annotation else ""
    text = re.sub(r'\s+', ' ', text).strip()
    # confidence (average)
    confidence = 0.0
    if annotation.pages:
        total_conf = 0
        total_symbols = 0
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            if symbol.confidence:
                                total_conf += symbol.confidence
                                total_symbols += 1
        if total_symbols > 0:
            confidence = (total_conf / total_symbols) * 100
    return text, confidence

def submit_pdf_to_vision_async(pdf_bytes):
    """Submit PDF to Vision async batch annotation, return operation name."""
    client = get_vision_async_client()
    # Save PDF to temp file (required by Vision API)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, 'rb') as f:
            content = f.read()
        input_config = types.InputConfig(
            mime_type='application/pdf',
            content=content
        )
        features = [types.Feature(type_=types.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        request = types.AsyncBatchAnnotateFilesRequest(
            requests=[types.AsyncAnnotateFileRequest(
                input_config=input_config,
                features=features
            )]
        )
        operation = client.async_batch_annotate_files(request=request)
        # Return operation name for polling
        return operation.operation.name
    finally:
        os.unlink(tmp_path)

def poll_vision_operation(operation_name):
    """Poll operation until done, return extracted text."""
    from google.cloud.vision_v1 import ImageAnnotatorClient
    client = get_vision_async_client()
    operation = client.transport.operations_client.get_operation(operation_name)
    if not operation.done:
        return None
    response = operation.response
    text_pages = []
    for result in response.responses:
        if result.output_config.gcs_destination_uri:
            # Not using GCS; we rely on inline output (not typical). 
            # Actually async response returns a list of AnnotateFileResponse.
            # We'll parse the response directly.
            pass
        # Simplified: iterate over responses
        for file_response in response.responses:
            for page in file_response.responses:
                if page.full_text_annotation:
                    text_pages.append(page.full_text_annotation.text)
    full_text = '\n\n'.join(text_pages)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text