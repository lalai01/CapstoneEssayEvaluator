import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import re
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import pytesseract
import os
import fitz  # PyMuPDF for PDF handling
import tempfile

# ----------------------------------------------------------------------
# Configure Tesseract path for Windows
# ----------------------------------------------------------------------
# Common installation paths for Tesseract on Windows
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]

tesseract_found = False
for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break

# ----------------------------------------------------------------------
# PDF Processing Functions
# ----------------------------------------------------------------------
def extract_images_from_pdf(pdf_path, dpi=300):
    """Extract images from PDF pages"""
    images = []
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Convert page to image
            zoom = dpi / 72  # 72 is default PDF resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            images.append(pil_image)
        
        pdf_document.close()
        return images
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

def extract_text_from_pdf_page(pil_image, page_num, preprocessing=True):
    """Extract text from a single PDF page image"""
    try:
        if preprocessing:
            pil_image = preprocess_image_for_ocr(pil_image)
        
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(pil_image, config=config)
        
        # Clean up text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        raise Exception(f"OCR failed on page {page_num}: {str(e)}")

def process_pdf_document(pdf_path):
    """Process entire PDF document and extract text from all pages"""
    try:
        # Extract images from PDF
        page_images = extract_images_from_pdf(pdf_path)
        
        all_text = []
        total_pages = len(page_images)
        
        for i, pil_image in enumerate(page_images):
            # Extract text from each page
            page_text = extract_text_from_pdf_page(pil_image, i + 1)
            if page_text:
                all_text.append(f"--- Page {i + 1} ---\n{page_text}")
        
        return "\n\n".join(all_text), total_pages
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

# ----------------------------------------------------------------------
# OCR Functions
# ----------------------------------------------------------------------
def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale if it's a PIL Image
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply sharpening filter
        image = image.filter(ImageFilter.SHARPEN)
        
        # Apply thresholding to get binary image
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
    
    return image

def extract_text_from_image(image_path, preprocessing=True):
    """
    Extract text from image using Tesseract OCR
    Returns extracted text and confidence score
    """
    try:
        # Open image
        image = Image.open(image_path)
        
        # Preprocess if requested
        if preprocessing:
            image = preprocess_image_for_ocr(image)
        
        # Perform OCR with multiple configuration options
        config = '--psm 6 --oem 3'
        
        # Get text
        text = pytesseract.image_to_string(image, config=config)
        
        # Get confidence data
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 75
        
        # Clean up text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text, avg_confidence
        
    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")

# ----------------------------------------------------------------------
# Essay Evaluation Functions
# ----------------------------------------------------------------------
def analyze_essay_content(essay_text):
    """Dynamically analyze essay content without static text"""
    words = essay_text.split()
    word_count = len(words)
    sentences = max(1, len(re.findall(r'[.!?]+', essay_text)))
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentences
    
    # Find unique words for vocabulary analysis
    unique_words = len(set(word.lower() for word in words))
    vocabulary_richness = (unique_words / word_count) if word_count > 0 else 0
    
    # Check for common transition words
    transition_words = ['however', 'therefore', 'consequently', 'furthermore', 'moreover', 
                       'nevertheless', 'subsequently', 'additionally', 'in conclusion']
    transition_count = sum(1 for word in words if word.lower() in transition_words)
    
    return {
        'word_count': word_count,
        'sentence_count': sentences,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'vocabulary_richness': vocabulary_richness,
        'transition_count': transition_count
    }

def calculate_grammar_score(essay_text, analysis):
    """Dynamically calculate grammar score based on content"""
    score = 85  # Base score
    
    # Adjust based on analysis
    if analysis['avg_sentence_length'] > 25:
        score -= 5  # Too many long sentences might indicate run-ons
    elif analysis['avg_sentence_length'] < 8:
        score -= 5  # Too many short, choppy sentences
    
    if analysis['vocabulary_richness'] < 0.4:
        score -= 5  # Low vocabulary diversity
    elif analysis['vocabulary_richness'] > 0.7:
        score += 3  # Good vocabulary diversity
    
    # Check for common grammar patterns (simplified)
    if re.search(r'\s+[,.!?]', essay_text):
        score -= 2  # Space before punctuation
    if re.search(r'[,.!?]{2,}', essay_text):
        score -= 3  # Multiple punctuation
    
    return max(60, min(98, score))

def calculate_coherence_score(essay_text, analysis):
    """Dynamically calculate coherence score"""
    score = 80  # Base score
    
    # Check for paragraph breaks
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        score += 5  # Good paragraph structure
    
    # Check transition words
    if analysis['transition_count'] > 3:
        score += 5
    elif analysis['transition_count'] == 0:
        score -= 5
    
    # Check for clear introduction and conclusion
    lower_text = essay_text.lower()
    intro_indicators = ['introduction', 'first', 'begin', 'start', 'purpose']
    conc_indicators = ['conclusion', 'summary', 'finally', 'in conclusion', 'to summarize']
    
    if any(word in lower_text[:500] for word in intro_indicators):
        score += 3
    if any(word in lower_text[-500:] for word in conc_indicators):
        score += 3
    
    return max(60, min(98, score))

def calculate_content_score(essay_text, analysis):
    """Dynamically calculate content score"""
    score = 75  # Base score
    
    # Longer essays tend to have more content
    if analysis['word_count'] > 500:
        score += 10
    elif analysis['word_count'] > 300:
        score += 5
    elif analysis['word_count'] < 100:
        score -= 10
    
    # Rich vocabulary suggests better content
    if analysis['vocabulary_richness'] > 0.6:
        score += 5
    
    # Check for supporting evidence indicators
    evidence_indicators = ['example', 'for instance', 'such as', 'because', 'research', 'study', 'data']
    evidence_count = sum(1 for indicator in evidence_indicators if indicator in essay_text.lower())
    score += min(10, evidence_count * 2)
    
    return max(60, min(98, score))

def generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type):
    """Generate completely dynamic feedback based on actual essay content"""
    
    feedback = []
    
    # Introduction based on essay length
    feedback.append("📊 **ESSAY ANALYSIS**")
    feedback.append(f"Your essay contains {analysis['word_count']} words across {analysis['sentence_count']} sentences.")
    feedback.append(f"Average sentence length: {analysis['avg_sentence_length']:.1f} words.")
    
    # Grammar feedback
    feedback.append("")
    feedback.append(f"📝 **GRAMMAR ANALYSIS (Score: {scores['grammar']})**")
    
    if scores['grammar'] < 70:
        grammar_issues = []
        if analysis['avg_sentence_length'] > 25:
            grammar_issues.append("- Some sentences are quite long. Consider breaking them into shorter, clearer sentences.")
        if analysis['vocabulary_richness'] < 0.4:
            grammar_issues.append("- Limited vocabulary detected. Try using more varied word choices.")
        if re.search(r'[,.!?][A-Za-z]', essay_text):
            grammar_issues.append("- Missing spaces after punctuation in some places.")
        
        if grammar_issues:
            feedback.extend(grammar_issues)
        else:
            feedback.append("- Several grammatical patterns need attention. Consider reviewing basic grammar rules.")
            
        feedback.append("")
        feedback.append("✅ **Suggestions for improvement:**")
        feedback.append("• Read your essay aloud to catch awkward phrasing")
        feedback.append("• Use grammar checking tools to identify specific errors")
        feedback.append("• Review subject-verb agreement in complex sentences")
        
    elif scores['grammar'] < 85:
        feedback.append("- Good basic grammar with room for refinement.")
        if analysis['transition_count'] < 3:
            feedback.append("- Adding more transition words would improve flow.")
        feedback.append("")
        feedback.append("✅ **Refinement suggestions:**")
        feedback.append("• Vary your sentence structure for better rhythm")
        feedback.append("• Check for consistent tense usage throughout")
    else:
        feedback.append("- Excellent grammar! Your writing is clear and correct.")
        feedback.append("")
        feedback.append("✅ **Advanced tips:**")
        feedback.append("• Experiment with more complex sentence structures")
        feedback.append("• Consider using stylistic devices for emphasis")
    
    # Coherence feedback
    feedback.append("")
    feedback.append(f"🔄 **COHERENCE ANALYSIS (Score: {scores['coherence']})**")
    
    paragraphs = essay_text.split('\n\n')
    if len(paragraphs) > 1:
        feedback.append(f"- Your essay has {len(paragraphs)} paragraphs, which is good for organization.")
    else:
        feedback.append("- Consider breaking your essay into paragraphs for better organization.")
    
    if analysis['transition_count'] > 0:
        feedback.append(f"- You used {analysis['transition_count']} transition words, which helps with flow.")
    else:
        feedback.append("- Adding transition words (however, therefore, moreover) would improve logical flow.")
    
    # Content feedback
    feedback.append("")
    feedback.append(f"📚 **CONTENT ANALYSIS (Score: {scores['content']})**")
    
    # Check essay structure
    words = essay_text.lower().split()
    if any(word in ['introduction', 'first', 'begin'] for word in words[:50]):
        feedback.append("- Good introduction detected.")
    else:
        feedback.append("- Consider adding a clearer introduction that states your main idea.")
    
    if any(word in ['conclusion', 'summary', 'finally'] for word in words[-50:]):
        feedback.append("- Good conclusion detected that wraps up your ideas.")
    else:
        feedback.append("- Add a conclusion that summarizes your main points.")
    
    # Evidence checking
    evidence_words = ['example', 'because', 'since', 'research', 'data', 'study', 'shows']
    evidence_count = sum(1 for word in evidence_words if word in essay_text.lower())
    if evidence_count > 3:
        feedback.append(f"- Good use of evidence and supporting details ({evidence_count} instances).")
    else:
        feedback.append("- Add more specific examples and evidence to support your arguments.")
    
    # Generate specific improvement example based on actual essay content
    if len(essay_text) > 100:
        # Take a sample sentence from the essay for improvement example
        sentences = re.split(r'[.!?]+', essay_text)
        sample_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
        
        if sample_sentences:
            sample = sample_sentences[0][:100] + "..." if len(sample_sentences[0]) > 100 else sample_sentences[0]
            feedback.append("")
            feedback.append("✨ **SAMPLE IMPROVEMENT BASED ON YOUR WRITING:**")
            feedback.append(f"Original: \"{sample}\"")
            
            # Generate enhanced version suggestion
            feedback.append("Enhanced version would include more specific details and varied vocabulary.")
            feedback.append("Tip: Try to replace general words with more precise terminology.")
    
    return "\n".join(feedback)

def evaluate_essay(essay_text, evaluation_type="analytic"):
    """Main evaluation function - completely dynamic, no static text"""
    
    # Analyze content
    analysis = analyze_essay_content(essay_text)
    
    # Calculate scores
    scores = {
        "grammar": calculate_grammar_score(essay_text, analysis),
        "coherence": calculate_coherence_score(essay_text, analysis),
        "content": calculate_content_score(essay_text, analysis)
    }
    
    # Generate feedback
    feedback = generate_dynamic_feedback(essay_text, scores, analysis, evaluation_type)
    
    return scores, feedback

# Simple in-memory knowledge base
knowledge_base = []

# Predefined rubric
RUBRIC = {
    "grammar": "Correctness of sentence structure, punctuation, and spelling.",
    "coherence": "Logical flow, transitions, and organization of ideas.",
    "content": "Depth, relevance, evidence, and quality of arguments."
}

# Suggestion guide
SUGGESTION_GUIDE = {
    "what": "This AI evaluates essays by dynamically analyzing your actual writing, providing personalized feedback based on your specific content.",
    "when": """Use this tool when you:
• Need constructive feedback on essay drafts
• Want to improve your writing skills
• Prepare for standardized tests (IELTS, TOEFL, GRE)
• Require consistent grading for multiple essays
• Want to digitize handwritten essays for analysis""",
    "how": """How to get the best results:
1. **Input Methods**:
   - Type or paste your essay directly
   - Upload an image (JPG, PNG, etc.) of handwritten/printed essay
   - Upload a PDF file (all pages will be processed)
   
2. **For Best OCR Results**:
   - Ensure good lighting when photographing
   - Use clear, legible handwriting
   - Avoid shadows and glare
   - For PDFs, ensure they are text-based or high-quality scans
   
3. **Review Process**:
   - Check extracted text for accuracy
   - Make manual corrections if needed
   - Then click Evaluate for personalized feedback"""
}

# ----------------------------------------------------------------------
# Main Application Class
# ----------------------------------------------------------------------
class EssayEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Essay Evaluator with OCR & PDF Support")
        self.root.geometry("1400x900")
        
        # Configure style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Suggestion.TLabelframe', font=('Arial', 11, 'bold'))

        # Variables
        self.essay_text = tk.StringVar()
        self.file_path = tk.StringVar()
        self.selected_model = tk.StringVar(value="Dynamic Analyzer")
        self.evaluation_type = tk.StringVar(value="analytic")
        self.ocr_confidence = tk.StringVar(value="Not available")
        self.ocr_method = tk.StringVar(value="Not used")
        self.teacher_satisfaction = tk.IntVar(value=5)
        self.file_type = tk.StringVar(value="None")
        
        # Image preview variables
        self.preview_image = None
        self.photo_image = None
        self.original_image = None
        self.current_pdf_images = []
        self.current_pdf_page = 0

        # Check Tesseract availability
        self.tesseract_available = tesseract_found
        
        # Build GUI
        self.create_widgets()
        self.show_welcome_message()
        
        # Show Tesseract status
        self.show_tesseract_status()

    def create_widgets(self):
        # Main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)

        # Left panel - Input and Preview (with scrollbar)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Add canvas with scrollbar for left panel
        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_scrollable = ttk.Frame(left_canvas)
        
        left_scrollable.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        self.setup_left_panel(left_scrollable)

        # Right panel - Notebook for results (with scrollbar)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Add canvas with scrollbar for right panel
        right_canvas = tk.Canvas(right_frame, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=right_canvas.yview)
        right_scrollable = ttk.Frame(right_canvas)
        
        right_scrollable.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        right_canvas.create_window((0, 0), window=right_scrollable, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")
        
        self.setup_right_panel(right_scrollable)

    def setup_left_panel(self, parent):
        # Use pack for the main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Input section
        input_frame = ttk.LabelFrame(main_container, text="Essay Input", padding=10)
        input_frame.pack(fill='both', expand=True, pady=5)

        # Essay type
        type_frame = ttk.Frame(input_frame)
        type_frame.pack(fill='x', pady=5)
        ttk.Label(type_frame, text="Essay Type:", style='Heading.TLabel').pack(side='left')
        ttk.Radiobutton(type_frame, text="Analytic", variable=self.evaluation_type, value="analytic").pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="Holistic", variable=self.evaluation_type, value="holistic").pack(side='left')

        # Text input
        ttk.Label(input_frame, text="Enter Essay Text:", style='Heading.TLabel').pack(anchor='w', pady=(10,5))
        self.essay_entry = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD, font=('Arial', 10))
        self.essay_entry.pack(fill='both', expand=True, pady=5)

        # File upload
        ttk.Label(input_frame, text="Upload File (Image or PDF):", style='Heading.TLabel').pack(anchor='w', pady=(10,5))
        
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill='x', pady=5)
        
        ttk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse", command=self.upload_file).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Extract Text", command=self.extract_text_from_file).pack(side='left', padx=2)

        # File type indicator
        ttk.Label(file_frame, textvariable=self.file_type, foreground='blue').pack(side='left', padx=10)

        # OCR Info
        ocr_frame = ttk.LabelFrame(input_frame, text="OCR Information", padding=5)
        ocr_frame.pack(fill='x', pady=10)
        
        info_line = ttk.Frame(ocr_frame)
        info_line.pack(fill='x')
        ttk.Label(info_line, text="Confidence:").pack(side='left')
        ttk.Label(info_line, textvariable=self.ocr_confidence, foreground='blue').pack(side='left', padx=5)
        ttk.Label(info_line, text="Method:").pack(side='left', padx=(20,0))
        ttk.Label(info_line, textvariable=self.ocr_method, foreground='green').pack(side='left', padx=5)

        # PDF Navigation (initially hidden)
        self.pdf_nav_frame = ttk.Frame(input_frame)
        self.pdf_nav_frame.pack(fill='x', pady=5)
        self.pdf_nav_frame.pack_forget()
        
        ttk.Button(self.pdf_nav_frame, text="◀ Previous", command=self.prev_pdf_page).pack(side='left', padx=5)
        self.pdf_page_label = ttk.Label(self.pdf_nav_frame, text="Page 0/0")
        self.pdf_page_label.pack(side='left', padx=10)
        ttk.Button(self.pdf_nav_frame, text="Next ▶", command=self.next_pdf_page).pack(side='left', padx=5)

        # Image/PDF preview
        self.preview_frame = ttk.LabelFrame(input_frame, text="File Preview", padding=5)
        self.preview_frame.pack(fill='both', expand=True, pady=10)
        
        # Add scrollbar for preview
        preview_scroll = ttk.Scrollbar(self.preview_frame)
        preview_scroll.pack(side='right', fill='y')
        
        self.preview_canvas = tk.Canvas(self.preview_frame, yscrollcommand=preview_scroll.set, height=200)
        self.preview_canvas.pack(side='left', fill='both', expand=True)
        preview_scroll.config(command=self.preview_canvas.yview)
        
        self.preview_label = ttk.Label(self.preview_canvas, text="No file loaded", relief='sunken', anchor='center')
        self.preview_canvas.create_window((0, 0), window=self.preview_label, anchor='nw')

        # Extracted text display
        self.extracted_frame = ttk.LabelFrame(input_frame, text="Extracted Text (from OCR)", padding=5)
        self.extracted_frame.pack(fill='both', expand=True, pady=10)
        
        # Add buttons for text management
        text_btn_frame = ttk.Frame(self.extracted_frame)
        text_btn_frame.pack(fill='x', pady=2)
        
        ttk.Button(text_btn_frame, text="Use for Evaluation", command=self.use_extracted_text).pack(side='left', padx=2)
        ttk.Button(text_btn_frame, text="Copy to Editor", command=self.copy_to_editor).pack(side='left', padx=2)
        ttk.Button(text_btn_frame, text="Clear", command=self.clear_extracted).pack(side='left', padx=2)
        
        self.extracted_text = scrolledtext.ScrolledText(self.extracted_frame, height=6, wrap=tk.WORD, font=('Arial', 9))
        self.extracted_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Evaluate button
        self.evaluate_btn = ttk.Button(input_frame, text="Evaluate Essay", command=self.start_evaluation, style='Title.TLabel')
        self.evaluate_btn.pack(pady=15)

        # Tesseract status
        self.status_frame = ttk.LabelFrame(input_frame, text="OCR Status", padding=5)
        self.status_frame.pack(fill='x', pady=5)

    def setup_right_panel(self, parent):
        # Use pack for right panel
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Notebook for tabs
        notebook = ttk.Notebook(container)
        notebook.pack(fill='both', expand=True)

        # Tab 1: Suggestions Guide
        suggestions_tab = ttk.Frame(notebook)
        notebook.add(suggestions_tab, text="💡 Suggestions Guide")
        self.setup_suggestions_tab(suggestions_tab)

        # Tab 2: Evaluation Results
        eval_tab = ttk.Frame(notebook)
        notebook.add(eval_tab, text="📊 Evaluation Results")
        self.setup_evaluation_tab(eval_tab)

        # Tab 3: Detailed Feedback
        feedback_tab = ttk.Frame(notebook)
        notebook.add(feedback_tab, text="📝 Detailed Feedback")
        self.setup_feedback_tab(feedback_tab)

        # Tab 4: Teacher Override
        override_tab = ttk.Frame(notebook)
        notebook.add(override_tab, text="✏️ Teacher Override")
        self.setup_override_tab(override_tab)

        # Tab 5: Knowledge Base
        kb_tab = ttk.Frame(notebook)
        notebook.add(kb_tab, text="📚 Knowledge Base")
        self.setup_kb_tab(kb_tab)

    def setup_suggestions_tab(self, parent):
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # What section
        what_frame = ttk.LabelFrame(scrollable_frame, text="❓ WHAT does this tool do?", padding=15)
        what_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(what_frame, text=SUGGESTION_GUIDE["what"], wraplength=600, justify='left').pack()

        # When section
        when_frame = ttk.LabelFrame(scrollable_frame, text="⏰ WHEN to use this tool?", padding=15)
        when_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(when_frame, text=SUGGESTION_GUIDE["when"], wraplength=600, justify='left').pack()

        # How section
        how_frame = ttk.LabelFrame(scrollable_frame, text="🔧 HOW to use this tool effectively?", padding=15)
        how_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(how_frame, text=SUGGESTION_GUIDE["how"], wraplength=600, justify='left').pack()

        # OCR Tips
        ocr_tips_frame = ttk.LabelFrame(scrollable_frame, text="📸 OCR TIPS FOR BEST RESULTS", padding=15)
        ocr_tips_frame.pack(fill='x', padx=10, pady=10)
        ocr_tips = """• Use high-resolution images (300 DPI or higher)
• Ensure text is straight (not skewed)
• Avoid shadows and glare on the page
• Use black text on white background for best results
• Handwriting should be clear and legible
• Crop image to remove non-text areas
• For PDFs, ensure they are text-based or high-quality scans
• PDFs with multiple pages will have all pages processed"""
        ttk.Label(ocr_tips_frame, text=ocr_tips, wraplength=600, justify='left').pack()

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_evaluation_tab(self, parent):
        # Use pack for evaluation tab
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Rubric display
        ttk.Label(container, text="Evaluation Rubric:", style='Heading.TLabel').pack(anchor='w')
        rubric_text = "\n".join([f"• {k.capitalize()}: {v}" for k, v in RUBRIC.items()])
        ttk.Label(container, text=rubric_text, justify='left').pack(anchor='w', padx=20, pady=5)

        # Scores frame
        score_frame = ttk.LabelFrame(container, text="Scores", padding=10)
        score_frame.pack(fill='x', pady=10)

        self.grammar_score = tk.StringVar(value="--")
        self.coherence_score = tk.StringVar(value="--")
        self.content_score = tk.StringVar(value="--")

        score_grid = ttk.Frame(score_frame)
        score_grid.pack()

        ttk.Label(score_grid, text="Grammar:", font=('Arial', 11, 'bold')).grid(row=0, column=0, padx=10)
        ttk.Label(score_grid, textvariable=self.grammar_score, font=('Arial', 14, 'bold'), 
                 foreground='blue').grid(row=0, column=1, padx=5)
        
        ttk.Label(score_grid, text="Coherence:", font=('Arial', 11, 'bold')).grid(row=0, column=2, padx=20)
        ttk.Label(score_grid, textvariable=self.coherence_score, font=('Arial', 14, 'bold'), 
                 foreground='green').grid(row=0, column=3, padx=5)
        
        ttk.Label(score_grid, text="Content:", font=('Arial', 11, 'bold')).grid(row=0, column=4, padx=20)
        ttk.Label(score_grid, textvariable=self.content_score, font=('Arial', 14, 'bold'), 
                 foreground='purple').grid(row=0, column=5, padx=5)

        # Summary
        ttk.Label(container, text="Summary:", style='Heading.TLabel').pack(anchor='w', pady=(10,5))
        self.summary_text = scrolledtext.ScrolledText(container, height=6, wrap=tk.WORD, font=('Arial', 10))
        self.summary_text.pack(fill='both', expand=True)

    def setup_feedback_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.feedback_display = scrolledtext.ScrolledText(container, wrap=tk.WORD, font=('Arial', 10))
        self.feedback_display.pack(fill='both', expand=True)

        # Configure tags for formatting
        self.feedback_display.tag_configure("heading", font=('Arial', 12, 'bold'), foreground='navy')

    def setup_override_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Accept/Reject buttons
        button_frame = ttk.Frame(container)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="✅ Accept Evaluation", command=self.accept_evaluation, width=20).pack(side='left', padx=5)
        ttk.Button(button_frame, text="❌ Reject & Override", command=self.reject_evaluation, width=20).pack(side='left', padx=5)

        # Override area
        self.override_frame = ttk.LabelFrame(container, text="Provide Corrected Feedback", padding=10)
        self.override_frame.pack(fill='both', expand=True, pady=10)
        
        ttk.Label(self.override_frame, text="Your feedback:").pack(anchor='w')
        self.override_text = scrolledtext.ScrolledText(self.override_frame, height=12, wrap=tk.WORD, font=('Arial', 10))
        self.override_text.pack(fill='both', expand=True, pady=5)
        
        ttk.Button(self.override_frame, text="Save Override to Knowledge Base", command=self.save_override).pack(pady=5)

        # Teacher satisfaction
        sat_frame = ttk.Frame(container)
        sat_frame.pack(fill='x', pady=5)
        ttk.Label(sat_frame, text="Teacher Satisfaction (1-10):").pack(side='left')
        ttk.Scale(sat_frame, from_=1, to=10, orient='horizontal', 
                 variable=self.teacher_satisfaction, length=200).pack(side='left', padx=10)
        ttk.Label(sat_frame, textvariable=self.teacher_satisfaction).pack(side='left')

        # Initially hide override frame
        self.override_frame.pack_forget()

    def setup_kb_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Search frame
        search_frame = ttk.Frame(container)
        search_frame.pack(fill='x', pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side='left')
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_kb())
        ttk.Entry(search_frame, textvariable=self.search_var, width=30).pack(side='left', padx=5)
        ttk.Button(search_frame, text="Refresh", command=self.refresh_kb).pack(side='right')

        # Knowledge base list
        list_frame = ttk.Frame(container)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.kb_listbox = tk.Listbox(list_frame, height=8, yscrollcommand=scrollbar.set, font=('Arial', 9))
        self.kb_listbox.pack(fill='both', expand=True)
        
        scrollbar.config(command=self.kb_listbox.yview)
        
        # Details display
        ttk.Label(container, text="Details:", style='Heading.TLabel').pack(anchor='w', pady=(10,5))
        self.kb_details = scrolledtext.ScrolledText(container, height=8, wrap=tk.WORD, font=('Arial', 9))
        self.kb_details.pack(fill='both', expand=True)
        
        # Bind selection
        self.kb_listbox.bind('<<ListboxSelect>>', self.show_kb_details)

    def show_tesseract_status(self):
        """Show Tesseract installation status"""
        if self.tesseract_available:
            status = "✓ Tesseract OCR is installed and ready"
            color = "green"
        else:
            status = "✗ Tesseract OCR not found. Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki"
            color = "red"
        
        # Clear previous status
        for widget in self.status_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(self.status_frame, text=status, foreground=color).pack()

    def show_welcome_message(self):
        """Show welcome message"""
        welcome_text = "Welcome to AI Essay Evaluator! Enter your essay text, upload an image, or upload a PDF file to begin."
        
        if hasattr(self, 'feedback_display'):
            self.feedback_display.insert('1.0', welcome_text)

    def upload_file(self):
        """Upload image or PDF file"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All supported files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.pdf"),
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            
            # Determine file type
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                self.file_type.set("📄 PDF Document")
                self.show_pdf_preview(file_path)
            else:
                self.file_type.set("🖼️ Image File")
                self.show_image_preview(file_path)
            
            # Hide PDF navigation initially
            self.pdf_nav_frame.pack_forget()

    def show_image_preview(self, image_path):
        """Display image preview"""
        try:
            # Load and resize image
            self.original_image = Image.open(image_path)
            pil_image = self.original_image.copy()
            
            # Calculate new size (max 400px width)
            max_width = 400
            if pil_image.width > max_width:
                ratio = max_width / pil_image.width
                new_height = int(pil_image.height * ratio)
                pil_image = pil_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Update preview
            self.preview_label.config(image=self.photo_image, text="")
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
            
            # Clear previous extracted text
            self.extracted_text.delete('1.0', tk.END)
            self.ocr_confidence.set("Ready to extract")
            self.ocr_method.set("Not started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def show_pdf_preview(self, pdf_path):
        """Display PDF preview (first page)"""
        try:
            # Extract first page as image
            pdf_images = extract_images_from_pdf(pdf_path, dpi=150)
            self.current_pdf_images = pdf_images
            self.current_pdf_page = 0
            
            if pdf_images:
                self.show_pdf_page(0)
                
                # Show PDF navigation if multiple pages
                if len(pdf_images) > 1:
                    self.pdf_nav_frame.pack(fill='x', pady=5, before=self.preview_frame)
                    self.update_pdf_page_label()
            
            # Clear previous extracted text
            self.extracted_text.delete('1.0', tk.END)
            self.ocr_confidence.set("Ready to extract")
            self.ocr_method.set("Not started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load PDF: {str(e)}")

    def show_pdf_page(self, page_num):
        """Show specific PDF page"""
        if 0 <= page_num < len(self.current_pdf_images):
            pil_image = self.current_pdf_images[page_num].copy()
            
            # Calculate new size
            max_width = 400
            if pil_image.width > max_width:
                ratio = max_width / pil_image.width
                new_height = int(pil_image.height * ratio)
                pil_image = pil_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(pil_image)
            
            # Update preview
            self.preview_label.config(image=self.photo_image, text="")
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def prev_pdf_page(self):
        """Go to previous PDF page"""
        if self.current_pdf_page > 0:
            self.current_pdf_page -= 1
            self.show_pdf_page(self.current_pdf_page)
            self.update_pdf_page_label()

    def next_pdf_page(self):
        """Go to next PDF page"""
        if self.current_pdf_page < len(self.current_pdf_images) - 1:
            self.current_pdf_page += 1
            self.show_pdf_page(self.current_pdf_page)
            self.update_pdf_page_label()

    def update_pdf_page_label(self):
        """Update PDF page label"""
        total = len(self.current_pdf_images)
        current = self.current_pdf_page + 1
        self.pdf_page_label.config(text=f"Page {current}/{total}")

    def extract_text_from_file(self):
        """Extract text from uploaded file (image or PDF)"""
        file_path = self.file_path.get()
        
        if not file_path:
            messagebox.showwarning("No File", "Please upload a file first.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found.")
            return
        
        # Disable button during processing
        self.evaluate_btn.config(state='disabled', text="Extracting Text...")
        
        # Run extraction in thread
        thread = threading.Thread(target=self.perform_extraction)
        thread.daemon = True
        thread.start()

    def perform_extraction(self):
        """Perform text extraction in background thread"""
        try:
            file_path = self.file_path.get()
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                # Process PDF
                if self.tesseract_available:
                    text, num_pages = process_pdf_document(file_path)
                    confidence = 75  # Approximate confidence for PDFs
                    method = f"Tesseract OCR (PDF, {num_pages} pages)"
                    
                    self.root.after(0, lambda: self.display_ocr_results(text, confidence, method))
                else:
                    self.root.after(0, lambda: messagebox.showerror("OCR Error", 
                        "Tesseract OCR is not installed. Please install it first."))
            else:
                # Process image
                if self.tesseract_available:
                    text, confidence = extract_text_from_image(file_path, preprocessing=True)
                    method = "Tesseract OCR (Image)"
                    
                    self.root.after(0, lambda: self.display_ocr_results(text, confidence, method))
                else:
                    self.root.after(0, lambda: messagebox.showerror("OCR Error", 
                        "Tesseract OCR is not installed. Please install it first."))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Extraction Error", str(e)))
        finally:
            self.root.after(0, lambda: self.evaluate_btn.config(state='normal', text="Evaluate Essay"))

    def display_ocr_results(self, text, confidence, method):
        """Display OCR results"""
        self.extracted_text.delete('1.0', tk.END)
        self.extracted_text.insert('1.0', text)
        
        self.ocr_confidence.set(f"{confidence:.1f}%")
        self.ocr_method.set(method)
        
        # Show word count
        word_count = len(text.split())
        char_count = len(text)
        
        messagebox.showinfo("Extraction Complete", 
                           f"Text extracted successfully!\n\n"
                           f"Words: {word_count}\n"
                           f"Characters: {char_count}\n"
                           f"Confidence: {confidence:.1f}%\n"
                           f"Method: {method}\n\n"
                           f"Please review the extracted text for accuracy.")

    def use_extracted_text(self):
        """Use extracted text for evaluation"""
        text = self.extracted_text.get('1.0', tk.END).strip()
        if text:
            self.essay_entry.delete('1.0', tk.END)
            self.essay_entry.insert('1.0', text)
            messagebox.showinfo("Success", "Extracted text copied to editor. You can now evaluate.")
        else:
            messagebox.showwarning("No Text", "No extracted text to use.")

    def copy_to_editor(self):
        """Copy extracted text to editor"""
        text = self.extracted_text.get('1.0', tk.END).strip()
        if text:
            self.essay_entry.delete('1.0', tk.END)
            self.essay_entry.insert('1.0', text)
        else:
            messagebox.showwarning("No Text", "No extracted text to copy.")

    def clear_extracted(self):
        """Clear extracted text"""
        self.extracted_text.delete('1.0', tk.END)
        self.ocr_confidence.set("Cleared")
        self.ocr_method.set("Cleared")

    def start_evaluation(self):
        # Disable button
        self.evaluate_btn.config(state='disabled', text="Evaluating...")
        
        # Run evaluation in thread
        thread = threading.Thread(target=self.run_evaluation)
        thread.daemon = True
        thread.start()

    def run_evaluation(self):
        # Get input
        essay = self.essay_entry.get("1.0", tk.END).strip()
        
        # If no essay but extracted text exists, offer to use it
        if not essay and hasattr(self, 'extracted_text'):
            extracted = self.extracted_text.get("1.0", tk.END).strip()
            if extracted:
                self.root.after(0, lambda: self.ask_use_extracted(extracted))
                self.root.after(0, self.reset_eval_button)
                return

        if not essay:
            self.root.after(0, lambda: messagebox.showerror("Error", "Please enter essay text."))
            self.root.after(0, self.reset_eval_button)
            return

        # Evaluate essay
        eval_type = self.evaluation_type.get()
        scores, feedback = evaluate_essay(essay, eval_type)

        # Update GUI with results
        self.root.after(0, lambda: self.display_results(essay, scores, feedback))
        self.root.after(0, self.reset_eval_button)

    def ask_use_extracted(self, extracted_text):
        """Ask user if they want to use extracted text"""
        if messagebox.askyesno("Use Extracted Text", 
                               "No text in editor but extracted text exists.\nUse extracted text for evaluation?"):
            self.essay_entry.delete('1.0', tk.END)
            self.essay_entry.insert('1.0', extracted_text)
            self.start_evaluation()

    def display_results(self, essay, scores, feedback):
        # Update scores
        self.grammar_score.set(str(scores["grammar"]))
        self.coherence_score.set(str(scores["coherence"]))
        self.content_score.set(str(scores["content"]))

        # Update summary
        self.summary_text.delete('1.0', tk.END)
        avg_score = (scores["grammar"] + scores["coherence"] + scores["content"]) // 3
        summary = f"Overall Score: {avg_score}/100\n"
        summary += f"Essay Length: {len(essay.split())} words\n"
        summary += f"Evaluation Type: {self.evaluation_type.get().capitalize()}"
        self.summary_text.insert('1.0', summary)

        # Update feedback display
        self.feedback_display.delete('1.0', tk.END)
        self.feedback_display.insert('1.0', feedback)

        # Store current evaluation
        self.current_evaluation = {
            "essay": essay,
            "grammar": scores["grammar"],
            "coherence": scores["coherence"],
            "content": scores["content"],
            "feedback": feedback,
            "eval_type": self.evaluation_type.get()
        }

    def reset_eval_button(self):
        self.evaluate_btn.config(state='normal', text="Evaluate Essay")

    def accept_evaluation(self):
        if hasattr(self, 'current_evaluation'):
            entry = self.current_evaluation.copy()
            entry["accepted"] = True
            entry["satisfaction"] = self.teacher_satisfaction.get()
            knowledge_base.append(entry)
            messagebox.showinfo("Accepted", "Evaluation accepted and stored in knowledge base.")
            self.refresh_kb()
        else:
            messagebox.showwarning("No Evaluation", "Please evaluate an essay first.")

    def reject_evaluation(self):
        if hasattr(self, 'current_evaluation'):
            self.override_frame.pack(fill='both', expand=True, padx=10, pady=10)
            self.override_text.delete('1.0', tk.END)
            self.override_text.insert('1.0', self.current_evaluation["feedback"])
        else:
            messagebox.showwarning("No Evaluation", "Please evaluate an essay first.")

    def save_override(self):
        if not hasattr(self, 'current_evaluation'):
            messagebox.showerror("Error", "No evaluation to override.")
            return

        teacher_fb = self.override_text.get("1.0", tk.END).strip()
        if not teacher_fb:
            messagebox.showwarning("Empty", "Please enter your corrected feedback.")
            return

        corrected = self.current_evaluation.copy()
        corrected["feedback"] = teacher_fb
        corrected["accepted"] = False
        corrected["satisfaction"] = self.teacher_satisfaction.get()
        knowledge_base.append(corrected)

        self.override_frame.pack_forget()
        messagebox.showinfo("Saved", "Your override has been saved to the knowledge base.")
        self.refresh_kb()

    def refresh_kb(self):
        self.kb_listbox.delete(0, tk.END)
        self.kb_storage = {}
        
        for i, item in enumerate(knowledge_base):
            display_text = f"Essay {i+1}: Score {item['grammar']}/{item['coherence']}/{item['content']} - {'✓' if item.get('accepted', False) else '✗'}"
            self.kb_listbox.insert(tk.END, display_text)
            self.kb_storage[i] = item

    def filter_kb(self):
        search_term = self.search_var.get().lower()
        self.kb_listbox.delete(0, tk.END)
        self.kb_storage = {}
        
        for i, item in enumerate(knowledge_base):
            if search_term in str(item).lower() or search_term in item.get('feedback', '').lower():
                display_text = f"Essay {i+1}: Score {item['grammar']}/{item['coherence']}/{item['content']} - {'✓' if item.get('accepted', False) else '✗'}"
                self.kb_listbox.insert(tk.END, display_text)
                self.kb_storage[len(self.kb_storage)] = item

    def show_kb_details(self, event):
        selection = self.kb_listbox.curselection()
        if selection and hasattr(self, 'kb_storage'):
            index = selection[0]
            if index in self.kb_storage:
                item = self.kb_storage[index]
                details = f"ESSAY:\n{item['essay'][:500]}...\n\n"
                details += f"SCORES: Grammar={item['grammar']}, Coherence={item['coherence']}, Content={item['content']}\n"
                details += f"TYPE: {item.get('eval_type', 'N/A')}\n"
                details += f"ACCEPTED: {'Yes' if item.get('accepted', False) else 'No (Teacher Override)'}\n"
                details += f"SATISFACTION: {item.get('satisfaction', 'N/A')}/10\n\n"
                details += f"FEEDBACK:\n{item['feedback']}"
                
                self.kb_details.delete('1.0', tk.END)
                self.kb_details.insert('1.0', details)

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EssayEvaluatorApp(root)
    root.mainloop()