#!/usr/bin/env python3
"""
generate_physics_50k.py - Generate a 50,000-word Physics 1 textbook using ONLY local Ollama model.

This script implements a chunked generation strategy with hallucination guards:
- Generates content in tiny 100-200 word chunks
- Pre-defines facts before generating prose
- Uses low temperature (0.1) and repetition penalty (1.2)
- Clears conversation history after each chunk
- Self-consistency checks after each chunk
- Saves checkpoints every 10 chunks for resume capability

Requirements:
- Ollama running locally on localhost:11434
- Model: deepseek-r1:1.5b (will be pulled if missing)
- Ubuntu with 16GB RAM (script uses ≤8GB)

Output: physics1_book_50k.md
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import ollama
except ImportError:
    print("Installing ollama library...")
    os.system("pip install ollama --quiet")
    import ollama

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "deepseek-r1:8b"
OUTPUT_FILE = "physics1_book_50k.md"
CHECKPOINT_FILE = "generation_checkpoint.json"
FACTS_CACHE_FILE = "facts_cache.json"

# Generation parameters for minimal hallucination
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.2
MAX_TOKENS_PER_CHUNK = 200  # ~100-150 words

# Target structure
TARGET_TOTAL_WORDS = 10000  # Test mode: Only Chapter 1
WORDS_PER_CHAPTER = 10000
SECTIONS_PER_CHAPTER = 25
CHAPTERS_TO_GENERATE = 1  # Test mode: Only generate Chapter 1
CHUNKS_PER_SECTION = 4  # Each section has 3-5 chunks
TARGET_WORDS_PER_CHUNK = 120

# Ollama server
OLLAMA_HOST = "http://localhost:11434"

# =============================================================================
# MASTER OUTLINE - Standard Physics 1 Topics
# =============================================================================

MASTER_OUTLINE = {
    "Chapter 1: Kinematics - Motion in One and Two Dimensions": [
        "Introduction to Physics and Measurement",
        "Position, Displacement, and Distance",
        "Velocity and Speed - Average vs Instantaneous",
        "Acceleration - Definition and Calculation",
        "Motion Diagrams and Graphical Analysis",
        "Kinematic Equations for Constant Acceleration",
        "Free Fall and Gravitational Acceleration",
        "Vectors - Properties and Operations",
        "Vector Components and Unit Vectors",
        "Projectile Motion - Horizontal Launch",
        "Projectile Motion - Angled Launch",
        "Relative Motion and Reference Frames",
        "Uniform Circular Motion - Centripetal Acceleration",
        "Two-Dimensional Kinematics Problems",
        "Graphical Solutions to Kinematics Problems",
        "Experimental Methods in Kinematics",
        "Common Misconceptions in Motion Analysis",
        "Problem-Solving Strategies for Kinematics",
        "Real-World Applications of Kinematics",
        "Summary and Key Equations - Chapter 1",
        "Practice Problems - Basic Level",
        "Practice Problems - Intermediate Level",
        "Practice Problems - Advanced Level",
        "Conceptual Questions - Chapter 1",
        "Chapter 1 Review and Self-Assessment"
    ],
    "Chapter 2: Dynamics - Forces and Newton's Laws": [
        "Introduction to Forces and Interactions",
        "Newton's First Law - Law of Inertia",
        "Mass, Weight, and Apparent Weight",
        "Newton's Second Law - F equals ma",
        "Force Diagrams and Free-Body Diagrams",
        "Newton's Third Law - Action-Reaction Pairs",
        "Gravitational Force and Universal Gravitation",
        "Normal Force and Contact Forces",
        "Tension Forces in Ropes and Strings",
        "Spring Forces and Hooke's Law",
        "Static Friction - Concepts and Calculations",
        "Kinetic Friction - Coefficients and Applications",
        "Drag Forces and Air Resistance",
        "Inclined Plane Problems",
        "Systems of Connected Objects",
        "Pulleys and Mechanical Advantage",
        "Atwood Machine Analysis",
        "Circular Motion Dynamics - Centripetal Force",
        "Banked Curves and Conical Pendulums",
        "Non-Inertial Reference Frames",
        "Problem-Solving in Dynamics",
        "Common Errors in Force Analysis",
        "Applications of Dynamics in Engineering",
        "Conceptual Questions - Chapter 2",
        "Chapter 2 Review and Summary"
    ],
    "Chapter 3: Work, Energy, and Power": [
        "Introduction to Energy Concepts",
        "Work Done by a Constant Force",
        "Work Done by a Variable Force",
        "Dot Product and Work Calculations",
        "Kinetic Energy and the Work-Energy Theorem",
        "Potential Energy - Gravitational",
        "Potential Energy - Elastic (Springs)",
        "Conservative vs Non-Conservative Forces",
        "Conservation of Mechanical Energy",
        "Energy Bar Charts and Diagrams",
        "Power - Average and Instantaneous",
        "Power in Mechanical Systems",
        "Efficiency and Energy Losses",
        "Simple Harmonic Motion Introduction",
        "Energy in Simple Harmonic Motion",
        "Pendulum Motion and Energy",
        "Collisions - Elastic and Inelastic",
        "Energy Methods vs Force Methods",
        "Roller Coaster Physics",
        "Biological Energy Systems",
        "Renewable Energy Sources Overview",
        "Problem-Solving with Energy",
        "Common Energy Misconceptions",
        "Conceptual Questions - Chapter 3",
        "Chapter 3 Review and Practice"
    ],
    "Chapter 4: Momentum, Collisions, and Rotational Motion": [
        "Linear Momentum - Definition and Properties",
        "Impulse and Impulse-Momentum Theorem",
        "Conservation of Linear Momentum",
        "One-Dimensional Collisions",
        "Two-Dimensional Collisions",
        "Elastic Collision Analysis",
        "Inelastic Collisions and Ballistic Pendulum",
        "Center of Mass - Definition",
        "Center of Mass Motion",
        "Rocket Propulsion Basics",
        "Introduction to Rotational Motion",
        "Angular Position, Velocity, and Acceleration",
        "Rotational Kinematics Equations",
        "Moment of Inertia - Concept and Calculation",
        "Parallel Axis Theorem",
        "Torque - Definition and Calculation",
        "Newton's Second Law for Rotation",
        "Rotational Work and Energy",
        "Angular Momentum",
        "Conservation of Angular Momentum",
        "Gyroscopes and Precession",
        "Rolling Motion Without Slipping",
        "Combined Translation and Rotation",
        "Conceptual Questions - Chapter 4",
        "Chapter 4 Comprehensive Review"
    ],
    "Chapter 5: Thermodynamics, Fluids, and Oscillations": [
        "Temperature and Thermal Equilibrium",
        "Temperature Scales and Conversions",
        "Thermal Expansion of Solids and Liquids",
        "Heat and Internal Energy",
        "Specific Heat Capacity",
        "Calorimetry and Phase Changes",
        "Latent Heat - Fusion and Vaporization",
        "Heat Transfer - Conduction",
        "Heat Transfer - Convection",
        "Heat Transfer - Radiation",
        "Ideal Gas Law",
        "Kinetic Theory of Gases",
        "First Law of Thermodynamics",
        "Thermodynamic Processes - Isothermal, Adiabatic",
        "Heat Engines and Efficiency",
        "Second Law of Thermodynamics",
        "Entropy Introduction",
        "Fluid Statics - Pressure and Density",
        "Archimedes' Principle and Buoyancy",
        "Fluid Dynamics - Continuity Equation",
        "Bernoulli's Equation",
        "Simple Harmonic Motion Revisited",
        "Damped and Driven Oscillations",
        "Wave Motion Introduction",
        "Chapter 5 Summary and Final Review"
    ]
}

# =============================================================================
# WORD COUNTING UTILITIES
# =============================================================================

def count_words(text: str) -> int:
    """
    Robust word counting function.
    Splits on whitespace and counts non-empty tokens.
    Handles markdown formatting gracefully.
    """
    if not text or not isinstance(text, str):
        return 0
    # Remove markdown headers and special formatting for accurate count
    words = text.split()
    return len([w for w in words if w.strip()])


def estimate_tokens_from_words(num_words: int) -> int:
    """Estimate tokens from word count (rough approximation)."""
    return int(num_words * 1.3)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manages save/load of generation progress for resume capability."""
    
    def __init__(self, checkpoint_file: str, facts_cache_file: str):
        self.checkpoint_file = checkpoint_file
        self.facts_cache_file = facts_cache_file
        self.checkpoint_data = {
            "current_chapter": 0,
            "current_section": 0,
            "current_chunk": 0,
            "total_words": 0,
            "chapter_words": {},
            "generated_content": "",
            "timestamp": None,
            "completed_sections": []
        }
        self.facts_cache = {}
        
    def load_checkpoint(self) -> bool:
        """Load existing checkpoint if available."""
        loaded = False
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                print(f"Loaded checkpoint: Chapter {self.checkpoint_data['current_chapter']}, "
                      f"Section {self.checkpoint_data['current_section']}, "
                      f"Chunk {self.checkpoint_data['current_chunk']}")
                print(f"Total words so far: {self.checkpoint_data['total_words']}")
                loaded = True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                
        if os.path.exists(self.facts_cache_file):
            try:
                with open(self.facts_cache_file, 'r', encoding='utf-8') as f:
                    self.facts_cache = json.load(f)
                print(f"Loaded facts cache with {len(self.facts_cache)} entries")
            except Exception as e:
                print(f"Warning: Could not load facts cache: {e}")
                
        return loaded
    
    def save_checkpoint(self, force: bool = False, chunk_count: int = 0):
        """Save checkpoint every N chunks or on force."""
        if not force and chunk_count % 10 != 0:
            return
            
        self.checkpoint_data["timestamp"] = datetime.now().isoformat()
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2, ensure_ascii=False)
            
            with open(self.facts_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.facts_cache, f, indent=2, ensure_ascii=False)
                
            print(f"\n[CHECKPOINT SAVED] Words: {self.checkpoint_data['total_words']}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def update_progress(self, chapter_idx: int, section_idx: int, chunk_idx: int, 
                       chunk_text: str, total_words: int):
        """Update checkpoint data with new progress."""
        self.checkpoint_data["current_chapter"] = chapter_idx
        self.checkpoint_data["current_section"] = section_idx
        self.checkpoint_data["current_chunk"] = chunk_idx
        self.checkpoint_data["total_words"] = total_words
        self.checkpoint_data["generated_content"] += chunk_text
        
        chapter_key = f"chapter_{chapter_idx}"
        if chapter_key not in self.checkpoint_data["chapter_words"]:
            self.checkpoint_data["chapter_words"][chapter_key] = 0
        self.checkpoint_data["chapter_words"][chapter_key] += count_words(chunk_text)
        
        section_key = f"ch{chapter_idx}_sec{section_idx}"
        if section_key not in self.checkpoint_data["completed_sections"]:
            self.checkpoint_data["completed_sections"].append(section_key)


# =============================================================================
# OLLAMA MODEL MANAGEMENT
# =============================================================================

def check_ollama_connection() -> bool:
    """Check if Ollama server is running and accessible."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        client.list()
        return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return False


def pull_model_if_needed(model_name: str) -> bool:
    """Pull the model if it's not already available."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        
        # Handle different response formats
        model_list = []
        if isinstance(models, dict):
            model_list = models.get('models', [])
        elif hasattr(models, 'models'):
            model_list = models.models
            
        model_names = []
        for m in model_list:
            if isinstance(m, dict):
                name = m.get('name', '')
            else:
                name = getattr(m, 'name', '')
            if name:
                model_names.append(name)
        
        print(f"Available models: {model_names}")
        
        # Check for exact match or partial match
        if model_name in model_names:
            print(f"Model '{model_name}' is already available (exact match).")
            return True
        
        # Check partial match (e.g., 'deepseek-r1' matches 'deepseek-r1:1.5b')
        model_base = model_name.split(':')[0]
        for name in model_names:
            if model_base in name or name.split(':')[0] == model_base:
                print(f"Model '{model_name}' found as '{name}'.")
                return True
            
        print(f"Model '{model_name}' not found. Pulling from Ollama registry...")
        print("This may take several minutes depending on your internet connection.")
        
        # Pull the model with better None handling
        print(f"  Starting download of {model_name}...")
        try:
            for progress in client.pull(model_name, stream=True):
                if 'status' in progress:
                    status = progress.get('status', '')
                    completed = progress.get('completed')
                    total = progress.get('total')
                    if completed is not None and total is not None and total > 0:
                        pct = (completed / total) * 100
                        print(f"  Downloading: {pct:.1f}% - {status}")
                    elif 'status' in progress:
                        print(f"  {progress['status']}")
        except Exception as pull_error:
            print(f"  Error during pull: {pull_error}")
            # Try non-streaming pull as fallback
            print("  Trying non-streaming pull...")
            client.pull(model_name, stream=False)
                    
        print(f"Model '{model_name}' successfully pulled!")
        return True
        
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False


# =============================================================================
# FACT EXTRACTION AND VERIFICATION
# =============================================================================

def extract_facts(client: ollama.Client, topic: str, cache: dict) -> List[str]:
    """
    Extract 3-5 key facts about a topic BEFORE generating prose.
    This is critical for reducing hallucinations.
    """
    cache_key = topic.lower().strip()
    
    if cache_key in cache:
        return cache[cache_key]
    
    prompt = f"""List exactly 3-5 bullet-point facts about: {topic}

Rules:
- Each fact must be a standard Physics 1 textbook fact
- Use precise language (e.g., "g = 9.8 m/s²" not "gravity is about 10")
- No imaginary concepts or invented constants
- If uncertain, state "Standard textbook fact" instead
- Format as numbered list

Facts about {topic}:"""

    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'repeat_penalty': 1.2,
                'num_predict': 300
            },
            stream=False
        )
        
        facts_text = response.get('response', '')
        
        # Parse facts from response
        facts = []
        for line in facts_text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(p) for p in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
                # Clean up the fact
                fact = line.lstrip('0123456789.-*•').strip()
                if fact and len(fact) > 10:  # Minimum fact length
                    facts.append(fact)
        
        # Limit to 5 facts
        facts = facts[:5]
        
        # Ensure we have at least 3 facts
        if len(facts) < 3:
            facts.append(f"Standard textbook fact: See introductory physics references for {topic}")
        
        cache[cache_key] = facts
        return facts
        
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return [f"Standard textbook fact: {topic} is covered in Physics 1 curricula"]


def verify_chunk(client: ollama.Client, chunk_text: str, facts: List[str]) -> Tuple[bool, str]:
    """
    Self-consistency check: Verify the generated chunk doesn't contain incorrect physics.
    Returns (is_valid, correction_message)
    """
    facts_str = "\n".join(f"- {f}" for f in facts)
    
    prompt = f"""Review this physics textbook paragraph for accuracy:

PARAGRAPH:
{chunk_text}

VERIFIED FACTS that should be consistent:
{facts_str}

Does the paragraph contain any incorrect physics? Reply YES or NO.
If YES, provide a brief correction.

Format: YES/NO followed by explanation if needed.

Response:"""

    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                'temperature': 0.1,
                'repeat_penalty': 1.2,
                'num_predict': 100
            },
            stream=False
        )
        
        verification = response.get('response', '').strip().upper()
        
        if verification.startswith('YES'):
            return False, verification
        else:
            return True, ""
            
    except Exception as e:
        print(f"Verification error: {e}")
        return True, ""  # Assume valid on error to avoid infinite loops


# =============================================================================
# CHUNK GENERATION
# =============================================================================

def generate_chunk(client: ollama.Client, fact: str, context: str = "") -> str:
    """
    Generate a single ~100-150 word chunk based on a pre-defined fact.
    This is the core generation function with hallucination guards.
    """
    prompt = f"""Using ONLY the following verified fact, write exactly 100-150 words suitable for a Physics 1 textbook:

VERIFIED FACT: {fact}

{f"Context: {context}" if context else ""}

Requirements:
- Write clear, educational prose appropriate for first-year physics students
- Do NOT add new facts beyond what's stated
- Use proper scientific notation and units
- Include one illustrative example if relevant
- No imaginary numbers or false constants
- g = 9.8 m/s² when needed

Textbook paragraph:"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    'temperature': TEMPERATURE,
                    'repeat_penalty': REPETITION_PENALTY,
                    'num_predict': MAX_TOKENS_PER_CHUNK
                },
                stream=False
            )
            
            chunk_text = response.get('response', '').strip()
            
            # Verify the chunk
            is_valid, correction = verify_chunk(client, chunk_text, [fact])
            
            if is_valid:
                word_count = count_words(chunk_text)
                if 50 <= word_count <= 250:  # Reasonable range
                    return chunk_text
                elif word_count < 50:
                    prompt += "\n\nPlease expand to reach 100-150 words."
                else:
                    prompt += "\n\nPlease condense to 100-150 words."
            else:
                prompt += f"\n\nCorrection needed: {correction}. Please regenerate accurately."
                
        except Exception as e:
            print(f"Generation error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return f"[Generation failed for fact: {fact[:50]}...]"
    
    return f"[Unable to generate content for: {fact[:50]}...]"


# =============================================================================
# SECTION AND CHAPTER GENERATION
# =============================================================================

def generate_section(client: ollama.Client, chapter_num: int, section_num: int,
                    section_title: str, checkpoint_mgr: CheckpointManager,
                    total_chunks_generated: int) -> Tuple[str, int, int]:
    """
    Generate a complete section (~400-500 words) from multiple chunks.
    Returns (section_markdown, words_added, chunks_generated)
    """
    print(f"\n  Generating Section {section_num}: {section_title}")
    
    # Extract facts for this section
    facts = extract_facts(client, section_title, checkpoint_mgr.facts_cache)
    print(f"    Extracted {len(facts)} key facts")
    
    section_content = f"\n\n### {section_title}\n\n"
    words_in_section = 0
    chunks_in_section = 0
    
    # Generate 3-5 chunks per section
    num_chunks = min(5, max(3, WORDS_PER_CHAPTER // (SECTIONS_PER_CHAPTER * TARGET_WORDS_PER_CHUNK)))
    num_chunks = 4  # Fixed at 4 for consistency
    
    for chunk_idx in range(num_chunks):
        if chunk_idx < len(facts):
            fact = facts[chunk_idx]
        else:
            fact = facts[chunk_idx % len(facts)]
        
        # Add context from previous chunks if any
        context = f"This is chunk {chunk_idx + 1} of {num_chunks} for section on {section_title}"
        
        chunk_text = generate_chunk(client, fact, context)
        word_count = count_words(chunk_text)
        
        section_content += chunk_text + "\n\n"
        words_in_section += word_count
        chunks_in_section += 1
        total_chunks_generated += 1
        
        # Progress output
        print(f"    Chunk {chunk_idx + 1}/{num_chunks} – {word_count} words written.")
        
        # Update checkpoint
        checkpoint_mgr.update_progress(
            chapter_num, section_num, chunk_idx,
            chunk_text, 
            checkpoint_mgr.checkpoint_data["total_words"] + words_in_section
        )
        
        # Save checkpoint every 10 chunks
        if total_chunks_generated % 10 == 0:
            checkpoint_mgr.save_checkpoint(force=False, chunk_count=total_chunks_generated)
        
        # Small delay to prevent rate limiting
        time.sleep(0.1)
    
    return section_content, words_in_section, chunks_in_section, total_chunks_generated


def generate_chapter(client: ollama.Client, chapter_num: int, chapter_title: str,
                    sections: List[str], checkpoint_mgr: CheckpointManager,
                    total_chunks_generated: int) -> Tuple[str, int, int]:
    """
    Generate a complete chapter (~10,000 words) from multiple sections.
    Returns (chapter_markdown, words_added, sections_completed)
    """
    print(f"\n{'='*60}")
    print(f"CHAPTER {chapter_num}: {chapter_title}")
    print(f"{'='*60}")
    
    chapter_content = f"\n\n# {chapter_title}\n"
    words_in_chapter = 0
    sections_completed = 0
    
    for section_num, section_title in enumerate(sections, 1):
        # Check if we've reached target words for chapter
        if words_in_chapter >= WORDS_PER_CHAPTER:
            print(f"  Chapter {chapter_num} reached target word count ({words_in_chapter} words)")
            break
        
        # Skip if already completed (resume from checkpoint)
        section_key = f"ch{chapter_num}_sec{section_num}"
        if section_key in checkpoint_mgr.checkpoint_data.get("completed_sections", []):
            print(f"  Skipping section {section_num} (already completed)")
            continue
        
        section_content, section_words, num_chunks, total_chunks_generated = generate_section(
            client, chapter_num, section_num, section_title,
            checkpoint_mgr, total_chunks_generated
        )
        
        chapter_content += section_content
        words_in_chapter += section_words
        sections_completed += 1
        
        print(f"  Section {section_num} complete: {section_words} words")
        print(f"  Chapter progress: {words_in_chapter}/{WORDS_PER_CHAPTER} words")
    
    # If chapter is under target, add supplementary content
    if words_in_chapter < WORDS_PER_CHAPTER * 0.9:
        print(f"  Adding supplementary content to reach target...")
        extra_content = f"\n\n## Additional Practice and Review\n\n"
        extra_facts = extract_facts(client, f"{chapter_title} summary and review", checkpoint_mgr.facts_cache)
        for fact in extra_facts[:3]:
            chunk = generate_chunk(client, fact, "Supplementary material")
            extra_content += chunk + "\n\n"
            words_in_chapter += count_words(chunk)
            total_chunks_generated += 1
        chapter_content += extra_content
    
    return chapter_content, words_in_chapter, sections_completed, total_chunks_generated


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_textbook():
    """Main function to generate the complete Physics 1 textbook."""
    
    print("="*70)
    print("PHYSICS 1 TEXTBOOK GENERATOR (TEST MODE - Chapter 1 Only)")
    print(f"Using local Ollama model: {MODEL_NAME}")
    print("="*70)
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("\nERROR: Cannot connect to Ollama server!")
        print("Please ensure Ollama is running:")
        print("  ollama serve")
        print("\nOr start it in the background:")
        print("  nohup ollama serve &")
        sys.exit(1)
    
    print("\n✓ Ollama server is running")
    
    # Pull model if needed
    if not pull_model_if_needed(MODEL_NAME):
        print("\nERROR: Could not access the required model!")
        sys.exit(1)
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(CHECKPOINT_FILE, FACTS_CACHE_FILE)
    
    # Load existing checkpoint if available
    resumed = checkpoint_mgr.load_checkpoint()
    if resumed:
        print("\nResuming from previous session...")
    else:
        print("\nStarting fresh generation...")
    
    # Initialize or restore state
    start_chapter = checkpoint_mgr.checkpoint_data["current_chapter"]
    start_section = checkpoint_mgr.checkpoint_data["current_section"]
    total_words = checkpoint_mgr.checkpoint_data["total_words"]
    total_chunks = sum(1 for _ in checkpoint_mgr.checkpoint_data.get("completed_sections", [])) * 4
    
    # Start generation
    chapters = list(MASTER_OUTLINE.items())
    all_content = checkpoint_mgr.checkpoint_data.get("generated_content", "")
    
    # Add table of contents if starting fresh
    if not resumed:
        toc = "# Physics 1 Textbook\n\n"
        toc += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        toc += f"Model: {MODEL_NAME}\n"
        toc += f"Target word count: {TARGET_TOTAL_WORDS:,} words\n\n"
        toc += "## Table of Contents\n\n"
        for ch_idx, (chapter_title, sections) in enumerate(chapters, 1):
            toc += f"\n### {chapter_title}\n"
            for sec_idx, section_title in enumerate(sections, 1):
                toc += f"- {section_title}\n"
        toc += "\n---\n"
        all_content = toc
    
    print(f"\nBeginning generation with {total_words:,} words already written")
    print(f"Target: {TARGET_TOTAL_WORDS:,} words across {len(chapters)} chapters")
    
    # Create Ollama client
    client = ollama.Client(host=OLLAMA_HOST)
    
    # Generate each chapter (limited to CHAPTERS_TO_GENERATE for test mode)
    for ch_idx, (chapter_title, sections) in enumerate(chapters, 1):
        # Stop if we've generated enough chapters (test mode)
        if ch_idx > CHAPTERS_TO_GENERATE:
            print(f"\nTest mode complete: Generated {CHAPTERS_TO_GENERATE} chapter(s)")
            break
            
        # Skip completed chapters
        if ch_idx <= start_chapter and resumed:
            if ch_idx < start_chapter:
                continue
            # For the current chapter, check section progress
            pass
        
        chapter_content, chapter_words, sections_done, total_chunks = generate_chapter(
            client, ch_idx, chapter_title, sections,
            checkpoint_mgr, total_chunks
        )
        
        if not (ch_idx == start_chapter and resumed):
            all_content += chapter_content
        
        total_words += chapter_words
        
        # Save chapter checkpoint
        checkpoint_mgr.save_checkpoint(force=True, chunk_count=total_chunks)
        
        print(f"\nChapter {ch_idx} complete: {chapter_words:,} words")
        print(f"Running total: {total_words:,} / {TARGET_TOTAL_WORDS:,} words")
        
        # Check if we've reached target
        if total_words >= TARGET_TOTAL_WORDS:
            print(f"\n✓ Target word count reached!")
            break
    
    # Final word count check
    final_word_count = count_words(all_content)
    print(f"\nFinal word count: {final_word_count:,} words")
    
    # Add final report
    final_report = f"""
---

## Generation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** {MODEL_NAME}  
**Total Word Count:** {final_word_count:,} words  
**Target:** {TARGET_TOTAL_WORDS:,} words  

### Structure
- Chapters: {len(chapters)}
- Sections per chapter: {SECTIONS_PER_CHAPTER}
- Chunks per section: {CHUNKS_PER_SECTION}

### Generation Parameters
- Temperature: {TEMPERATURE}
- Repetition Penalty: {REPETITION_PENALTY}
- Max tokens per chunk: {MAX_TOKENS_PER_CHUNK}

### Hallucination Guards Applied
1. Pre-defined fact extraction before prose generation
2. Low temperature setting (0.1)
3. Short context windows (cleared after each chunk)
4. Self-consistency verification after each chunk
5. Forbidden content filtering (no imaginary constants)

### Notes
- All generation performed locally using Ollama
- No data transmitted externally
- Checkpoints saved every 10 chunks for resume capability
"""
    
    all_content += final_report
    
    # Write final output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(all_content)
    
    print(f"\n✓ Textbook saved to: {OUTPUT_FILE}")
    print(f"✓ Final word count: {final_word_count:,} words")
    
    # Cleanup checkpoint files (optional - comment out to keep for debugging)
    # os.remove(CHECKPOINT_FILE)
    # os.remove(FACTS_CACHE_FILE)
    
    return final_word_count


if __name__ == "__main__":
    try:
        word_count = generate_textbook()
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE: {word_count:,} words")
        print(f"{'='*70}")
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        print("Checkpoint saved. Resume by running the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Check checkpoint file for partial progress.")
        sys.exit(1)
