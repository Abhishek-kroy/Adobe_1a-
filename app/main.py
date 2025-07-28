import os
import json
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer, util
import argparse
import re
import torch
from collections import defaultdict

# Initialize the sentence transformer model
model = SentenceTransformer("localmodel")

def clean_spacing(text):
    """Normalize whitespace in text"""
    return re.sub(r'\s+', ' ', text.strip())

def is_header_footer(text, page_num):
    """Identify header/footer content"""
    text_lower = text.lower().strip()
    patterns = [
        r'^\d+$', r'^page\s*\d+$', r'copyright\s+\d{4}', 
        r'^www\.', r'@\w+\.\w+', r'^confidential',
        r'^proprietary', r'^draft', r'^section\s+\d+',
        r'^appendix\s+[a-z]', r'^table\s+of\s+contents'
    ]
    return any(re.search(p, text_lower) for p in patterns)

def is_likely_heading(text, font_size, avg_font_size, max_font_size):
    """Determine if text is likely a heading"""
    if len(text) < 3 or len(text) > 150:
        return False
    
    text_lower = text.lower().strip()
    skip_patterns = [
        r'^\d+$', r'^page \d+', r'^chapter \d+', 
        r'^figure \d+', r'^table \d+', r'^source:',
        r'^note:', r'copyright', r'^[a-z][a-z\s]{0,10}$'
    ]
    
    if any(re.search(p, text_lower) for p in skip_patterns):
        return False
    
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False
    
    word_count = len(text.split())
    if word_count > 10:
        return False
    
    font_ratio = font_size / max_font_size if max_font_size > 0 else 0.5
    is_large_font = font_size > avg_font_size * 1.2 or font_ratio > 0.75
    
    indicators = [
        text.isupper() and word_count <= 6,
        bool(re.match(r'^\d+\.?\s+[A-Z]', text)),
        bool(re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z])$', text)),
        word_count <= 6,
        text.endswith(':'),
        bool(re.match(r'^[IVX]+\.?\s+', text)),
        bool(re.match(r'^[A-Z]\.?\s+', text)),
        bool(re.match(r'^(appendix|section|chapter)\s+', text_lower)),
    ]
    
    return is_large_font or sum(indicators) >= 2

def extract_lines(pdf_path, header_footer_margin=0.12):
    """Extract text lines from PDF with positioning info"""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_h = page.height
            header_cut = page_h * header_footer_margin
            footer_cut = page_h * (1 - header_footer_margin)

            words = page.extract_words(
                extra_attrs=["size", "top", "x0", "x1"],
                keep_blank_chars=False,
                x_tolerance=1,
                y_tolerance=1,
                use_text_flow=True
            )
            
            grouped = defaultdict(list)
            for w in words:
                if w["top"] < header_cut or w["top"] > footer_cut:
                    continue
                grouped[round(w["top"])].append(w)

            for y_pos, line_words in grouped.items():
                line_words.sort(key=lambda x: x["x0"])
                line_text = ""
                prev_x1 = None
                
                for i, word in enumerate(line_words):
                    current_text = word["text"]
                    if i > 0:
                        gap = word["x0"] - prev_x1
                        prev_word = line_words[i-1]
                        prev_width = prev_word["x1"] - prev_word["x0"]
                        avg_char_width = prev_width / max(len(prev_word["text"]), 1)
                        if gap > avg_char_width * 0.4:
                            line_text += " "
                    line_text += current_text
                    prev_x1 = word["x1"]
                
                text = clean_spacing(line_text)
                if len(text) < 3 or is_header_footer(text, page_num):
                    continue
                
                lines.append({
                    "text": text,
                    "font_size": np.mean([w["size"] for w in line_words]),
                    "page": page_num,
                    "top": y_pos
                })
    
    return lines

def calculate_heading_score(line, max_font_size):
    """Calculate importance score for a heading candidate"""
    font_score = line["font_size"] / max_font_size
    text = line["text"]
    text_score = 0
    
    # Text features that increase importance
    if text.isupper():
        text_score += 0.2
    if text.endswith(':'):
        text_score += 0.1
    if re.match(r'^\d+\.', text):
        text_score += 0.1
    if len(text.split()) <= 5:
        text_score += 0.1
    if re.match(r'^(section|chapter|appendix)\b', text, re.I):
        text_score += 0.2
    
    return 0.6 * font_score + 0.4 * text_score

def filter_headings(lines):
    """Select top 4 most important headings per page"""
    if not lines:
        return []
    
    # Group by page and process each page separately
    page_groups = defaultdict(list)
    for line in lines:
        page_groups[line["page"]].append(line)
    
    headings = []
    for page, page_lines in page_groups.items():
        # Calculate font statistics for current page
        font_sizes = [l["font_size"] for l in page_lines]
        max_font_size = max(font_sizes) if font_sizes else 1
        
        # Score and filter headings
        scored_headings = []
        for line in page_lines:
            if not is_likely_heading(line["text"], line["font_size"], 
                                   np.mean(font_sizes), max_font_size):
                continue
                
            score = calculate_heading_score(line, max_font_size)
            scored_headings.append((score, line))
        
        # Sort by score and take top 4
        scored_headings.sort(reverse=True, key=lambda x: x[0])
        top_headings = [line for (score, line) in scored_headings[:4]]
        
        # Clean and add to final headings
        for heading in top_headings:
            heading["text"] = clean_spacing(heading["text"])
            headings.append(heading)
    
    return headings

def hierarchical_parser(lines, sim_thr=(0.75, 0.60)):
    """Create hierarchical outline from selected headings"""
    if not lines:
        return {"title": "", "headings": []}
    
    # Select title from top candidates on first page
    first_page = [line for line in lines if line["page"] == 1][:4]
    title = max(first_page, key=lambda x: x["font_size"])["text"] if first_page else ""
    
    # Process headings hierarchy
    max_font = max(line["font_size"] for line in lines) if lines else 1
    headings = []
    context = []
    
    for i, line in enumerate(lines):
        text = line["text"]
        emb = model.encode(text, convert_to_tensor=True)
        
        if i == 0:
            level = "H1"
        else:
            # Calculate similarity with previous headings
            if context:
                sim_scores = util.cos_sim(emb, torch.stack(context))
                best_sim = torch.max(sim_scores).item()
            else:
                best_sim = 0
            
            size_ratio = line["font_size"] / max_font
            combined_score = 0.6 * best_sim + 0.4 * size_ratio
            
            if combined_score >= sim_thr[0] and size_ratio > 0.8:
                level = "H1"
            elif combined_score >= sim_thr[1] and size_ratio > 0.65:
                level = "H2"
            else:
                level = "H3"
        
        headings.append({
            "level": level,
            "text": text,
            "page": line["page"]
        })
        
        # Maintain context of recent headings
        context.append(emb)
        if len(context) > 3:
            context.pop(0)
    
    return {"title": title, "headings": headings}

def build_outline_json(pdf_path, output_path="semantic_outline.json", margin=0.12):
    """Main function to process PDF and generate outline"""
    print(f"Processing PDF: {pdf_path}")
    
    # Extract and process text lines
    lines = extract_lines(pdf_path, margin)
    if not lines:
        print("No text lines found in the PDF")
        return
    
    # Filter to get most important headings (max 4 per page)
    headings = filter_headings(lines)
    if not headings:
        print("No headings detected in the PDF")
        return
    
    # Create hierarchical outline
    result = hierarchical_parser(headings)
    
    # Save results
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
# ────────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"

    # Find the first PDF file in input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF file found in app/input/")
        exit(1)

    pdf_filename = pdf_files[0]
    pdf_path = os.path.join(input_dir, pdf_filename)
    
    # Generate output path with .json extension
    output_filename = os.path.splitext(pdf_filename)[0] + ".json"
    output_path = os.path.join(output_dir, output_filename)

    # Build the outline
    build_outline_json(pdf_path, output_path)
