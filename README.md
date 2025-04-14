<!-- omit in toc -->
# From Unstructured Text to Interactive Knowledge Graphs Using LLMs

<!-- omit in toc -->
# Table of Contents
- [Theory: What is a Knowledge Graph?](#theory-what-is-a-knowledge-graph)
- [Theory: Subject-Predicate-Object (SPO) Triples](#theory-subject-predicate-object-spo-triples)
- [Step 1: Setup - Installing Libraries](#step-1-setup---installing-libraries)
- [Step 2: Setup - Importing Libraries](#step-2-setup---importing-libraries)
- [Step 3: Configure LLM Access](#step-3-configure-llm-access)
- [Step 4: Define Input Text](#step-4-define-input-text)
- [Step 5: Text Chunking (Optional but Recommended)](#step-5-text-chunking-optional-but-recommended)
- [Step 6: Define the LLM Prompt for Extraction](#step-6-define-the-llm-prompt-for-extraction)
- [Step 7: LLM Interaction - Extracting Triples (Chunk by Chunk)](#step-7-llm-interaction---extracting-triples-chunk-by-chunk)
  - [Processing Chunk 1 (Example - loop structure will handle all)](#processing-chunk-1-example---loop-structure-will-handle-all)
  - [Extraction Summary (After Processing All Chunks)](#extraction-summary-after-processing-all-chunks)
- [Step 8: Normalize and De-duplicate Triples](#step-8-normalize-and-de-duplicate-triples)
- [Step 9: Build the Knowledge Graph with NetworkX](#step-9-build-the-knowledge-graph-with-networkx)
- [Step 10: Visualize the Graph Interactively with ipycytoscape](#step-10-visualize-the-graph-interactively-with-ipycytoscape)
  - [10.1 Convert NetworkX Data to Cytoscape Format](#101-convert-networkx-data-to-cytoscape-format)
  - [10.2 Create and Configure the Cytoscape Widget](#102-create-and-configure-the-cytoscape-widget)
  - [10.3 Define Visual Style](#103-define-visual-style)
  - [10.4 Set Layout Algorithm](#104-set-layout-algorithm)
  - [10.5 Display the Interactive Graph](#105-display-the-interactive-graph)
- [Step 11: Conclusion and Next Steps](#step-11-conclusion-and-next-steps)

**Goal:** This notebook demonstrates a **highly granular, step-by-step process** to transform raw, unstructured text into a structured, interactive knowledge graph using Large Language Models (LLMs). We will extract factual information (SPO triples) and visualize the data transformations and final graph **directly within the notebook** at multiple stages.

**Target Audience:** Beginners to Intermediate Python users interested in NLP, Knowledge Graphs, and LLMs, who want to see the data evolve at each step.

**Approach:** We will break down the process into very small, logical steps. Each step will aim to perform a distinct action, followed by an output or visualization to show the immediate result. We'll use basic Python constructs and popular libraries, prioritizing clarity and step-by-step understanding over code conciseness.

## Theory: What is a Knowledge Graph?

A Knowledge Graph (KG) is a way to represent information as a network of entities and their relationships. Think of it like a structured database, but instead of tables, you have:

*   **Nodes (or Entities):** These represent real-world objects, concepts, people, places, organizations, etc. (e.g., 'Marie Curie', 'Physics', 'Paris'). In our graph, each unique subject or object from our extracted facts will become a node.
*   **Edges (or Relationships):** These represent the connections or interactions between entities. They typically have a direction and a label describing the relationship (e.g., 'Marie Curie' -- `won` --> 'Nobel Prize', 'Radium' -- `is element discovered by` --> 'Marie Curie'). In our graph, each predicate from our extracted facts defines an edge between the corresponding subject and object nodes.

Knowledge graphs make it easier to understand complex connections, infer new information, and query data in intuitive ways. Visualizing the graph helps immensely in spotting patterns and understanding the overall structure.

## Theory: Subject-Predicate-Object (SPO) Triples

The fundamental building block of many knowledge graphs derived from text is the **Subject-Predicate-Object (SPO)** triple. It's a simple structure that captures a single fact:

*   **Subject:** The entity the statement is about (becomes a node).
*   **Predicate:** The relationship or action connecting the subject and object (becomes the label on an edge).
*   **Object:** The entity related to the subject via the predicate (becomes another node).

**Example:** "Marie Curie discovered Radium" -> (`Marie Curie`, `discovered`, `Radium`).

This translates to graph nodes and edges: `(Marie Curie) -[discovered]-> (Radium)`.

LLMs help identify these triples by understanding language context.

## Step 1: Setup - Installing Libraries

First, we install the necessary Python libraries. We'll use:
*   `openai`: For LLM API interaction.
*   `networkx`: For graph data structures.
*   `ipycytoscape`: For interactive in-notebook graph visualization.
*   `ipywidgets`: Required by `ipycytoscape`.
*   `pandas`: For displaying data nicely in tables.

**Note:** You might need to restart the runtime/kernel after installation. Enable `ipywidgets` extension in classic Jupyter Notebook if needed.


```python
# Install libraries (run this cell once)
%pip install openai networkx "ipycytoscape>=1.3.1" ipywidgets pandas

# If in classic Jupyter Notebook (not Lab), you might need to enable the widget extension:
# jupyter nbextension enable --py widgetsnbextension

# --- IMPORTANT: Restart the kernel/runtime after running this cell! ---
```

## Step 2: Setup - Importing Libraries

Now that the libraries are installed, we import the necessary components into our Python environment.


```python
import openai             # For LLM interaction
import json               # For parsing LLM responses
import networkx as nx     # For creating and managing the graph data structure
import ipycytoscape       # For interactive in-notebook graph visualization
import ipywidgets         # For interactive elements
import pandas as pd       # For displaying data in tables
import os                 # For accessing environment variables (safer for API keys)
import math               # For basic math operations
import re                 # For basic text cleaning (regular expressions)
import warnings           # To suppress potential deprecation warnings

# Configure settings for better display and fewer warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.set_option('display.max_rows', 100) # Show more rows in pandas tables
pd.set_option('display.max_colwidth', 150) # Show more text width in pandas tables

print("Libraries imported successfully.")
```

    Libraries imported successfully.
    

## Step 3: Configure LLM Access

We need to specify how to connect to the Large Language Model. This involves the API endpoint (URL) and the API key.

**IMPORTANT SECURITY NOTE:** Use environment variables or a secure secrets manager for API keys. **Do not hardcode keys directly in the notebook.**

**Environment Variable Setup (Example - run in your terminal *before* starting Jupyter):**
```bash
# For OpenAI
export OPENAI_API_KEY='your_openai_api_key'

# For Ollama (example)
export OPENAI_API_KEY='ollama' # Or any non-empty string
export OPENAI_API_BASE='http://localhost:11434/v1'

# For Nebius AI (example)
export OPENAI_API_KEY='your_nebius_api_key'
export OPENAI_API_BASE='https://api.studio.nebius.com/v1/'
```
First, we'll define the model name we intend to use.


```python
# --- Define LLM Model --- 
# Choose the model available at your configured endpoint.
# Examples: 'gpt-4o', 'gpt-3.5-turbo', 'llama3', 'mistral', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'gemma'
llm_model_name = "deepseek-ai/DeepSeek-V3" # <-- *** CHANGE THIS TO YOUR MODEL ***

print(f"Intended LLM model: {llm_model_name}")
```

    Intended LLM model: deepseek-ai/DeepSeek-V3
    

Now, let's retrieve the API key and base URL from environment variables.


```python
# --- Retrieve Credentials --- 
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE") # Will be None if not set (e.g., for standard OpenAI)

# --- FOR TESTING ONLY (Less Secure - Replace with Environment Variables) --- 
# Uncomment and set these lines ONLY if you cannot set environment variables easily.
# api_key = "YOUR_API_KEY_HERE"  # <--- PASTE KEY HERE FOR TESTING ONLY
# base_url = "YOUR_API_BASE_URL_HERE" # <--- PASTE BASE URL HERE (if needed)
# Example for Nebius:
# base_url="https://api.studio.nebius.com/v1/"
# api_key="YOUR_NEBIUS_KEY"

print(f"Retrieved API Key: {'Set' if api_key else 'Not Set'}")
print(f"Retrieved Base URL: {base_url if base_url else 'Not Set (will use default OpenAI)'}")
```

    Retrieved API Key: Set
    Retrieved Base URL: https://api.studio.nebius.com/v1/
    

Next, we validate the API key and initialize the `openai` client.


```python
# --- Validate Key and Initialize Client --- 
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set or key not provided directly.")
    print("Please set the environment variable (or uncomment/edit the test lines) and restart the kernel.")
    raise SystemExit("API Key configuration failed.")
else:
    try:
        client = openai.OpenAI(
            base_url=base_url, # Pass None if not set, client handles default
            api_key=api_key
        )
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Check your API key, base URL (if used), and network connection.")
        raise SystemExit("LLM client initialization failed.")
```

    OpenAI client initialized successfully.
    

Finally, let's define other LLM parameters like temperature and max tokens.


```python
# --- Define LLM Call Parameters ---
llm_temperature = 0.0 # Lower temperature for more deterministic, factual output. 0.0 is best for extraction.
llm_max_tokens = 4096 # Max tokens for the LLM response (adjust based on model limits)

print(f"LLM Temperature set to: {llm_temperature}")
print(f"LLM Max Tokens set to: {llm_max_tokens}")
```

    LLM Temperature set to: 0.0
    LLM Max Tokens set to: 4096
    

## Step 4: Define Input Text

Here, we define the raw, unstructured text we want to process. We'll use the Marie Curie biography.


```python
unstructured_text = """
Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
in two different scientific fields. She won the Nobel Prize in Physics in 1903 with Pierre Curie
and Henri Becquerel. Later, she won the Nobel Prize in Chemistry in 1911 for her work on radium and
polonium. During World War I, she developed mobile radiography units, known as 'petites Curies',
to provide X-ray services to field hospitals. Marie Curie died in 1934 from aplastic anemia, likely
caused by her long-term exposure to radiation.

Marie was born on November 7, 1867, to a family of teachers who valued education. She received her
early schooling in Warsaw but moved to Paris in 1891 to continue her studies at the Sorbonne, where
she earned degrees in physics and mathematics. She met Pierre Curie, a professor of physics, in 1894, 
and they married in 1895, beginning a productive scientific partnership. Following Pierre's tragic 
death in a street accident in 1906, Marie took over his teaching position, becoming the first female 
professor at the Sorbonne.

The Curies' work on radioactivity was conducted in challenging conditions, in a poorly equipped shed 
with no proper ventilation, as they processed tons of pitchblende ore to isolate radium. Marie Curie
established the Curie Institute in Paris, which became a major center for medical research. She had
two daughters: Irène, who later won a Nobel Prize in Chemistry with her husband, and Eve, who became
a writer. Marie's notebooks are still radioactive today and are kept in lead-lined boxes. Her legacy
includes not only her scientific discoveries but also her role in breaking gender barriers in academia
and science.
"""
```

Let's display the input text and some basic statistics about it.


```python
print("--- Input Text Loaded ---")
print(unstructured_text)
print("-" * 25)
# Basic stats visualization
char_count = len(unstructured_text)
word_count = len(unstructured_text.split())
print(f"Total characters: {char_count}")
print(f"Approximate word count: {word_count}")
print("-" * 25)
```

    --- Input Text Loaded ---
    
    Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
    She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
    she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
    the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
    in two different scientific fields. She won the Nobel Prize in Physics in 1903 with Pierre Curie
    and Henri Becquerel. Later, she won the Nobel Prize in Chemistry in 1911 for her work on radium and
    polonium. During World War I, she developed mobile radiography units, known as 'petites Curies',
    to provide X-ray services to field hospitals. Marie Curie died in 1934 from aplastic anemia, likely
    caused by her long-term exposure to radiation.
    
    Marie was born on November 7, 1867, to a family of teachers who valued education. She received her
    early schooling in Warsaw but moved to Paris in 1891 to continue her studies at the Sorbonne, where
    she earned degrees in physics and mathematics. She met Pierre Curie, a professor of physics, in 1894, 
    and they married in 1895, beginning a productive scientific partnership. Following Pierre's tragic 
    death in a street accident in 1906, Marie took over his teaching position, becoming the first female 
    professor at the Sorbonne.
    
    The Curies' work on radioactivity was conducted in challenging conditions, in a poorly equipped shed 
    with no proper ventilation, as they processed tons of pitchblende ore to isolate radium. Marie Curie
    established the Curie Institute in Paris, which became a major center for medical research. She had
    two daughters: Irène, who later won a Nobel Prize in Chemistry with her husband, and Eve, who became
    a writer. Marie's notebooks are still radioactive today and are kept in lead-lined boxes. Her legacy
    includes not only her scientific discoveries but also her role in breaking gender barriers in academia
    and science.
    
    -------------------------
    Total characters: 1995
    Approximate word count: 324
    -------------------------
    

## Step 5: Text Chunking (Optional but Recommended)

LLMs have context limits. For longer texts, we need to break them into smaller chunks. We'll define the chunk size and overlap.

*   **Chunk Size:** Max words per chunk.
*   **Overlap:** Words shared between consecutive chunks to preserve context.


```python
# --- Chunking Configuration ---
chunk_size = 150  # Number of words per chunk (adjust as needed)
overlap = 30     # Number of words to overlap (must be < chunk_size)

print(f"Chunk Size set to: {chunk_size} words")
print(f"Overlap set to: {overlap} words")

# --- Basic Validation ---
if overlap >= chunk_size and chunk_size > 0:
    print(f"Error: Overlap ({overlap}) must be smaller than chunk size ({chunk_size}).")
    raise SystemExit("Chunking configuration error.")
else:
    print("Chunking configuration is valid.")
```

    Chunk Size set to: 150 words
    Overlap set to: 30 words
    Chunking configuration is valid.
    

First, let's split the input text into a list of words.


```python
words = unstructured_text.split()
total_words = len(words)

print(f"Text split into {total_words} words.")
# Visualize the first 20 words
print(f"First 20 words: {words[:20]}")
```

    Text split into 324 words.
    First 20 words: ['Marie', 'Curie,', 'born', 'Maria', 'Skłodowska', 'in', 'Warsaw,', 'Poland,', 'was', 'a', 'pioneering', 'physicist', 'and', 'chemist.', 'She', 'conducted', 'groundbreaking', 'research', 'on', 'radioactivity.']
    

Now, we'll perform the chunking based on the configuration.


```python
chunks = []
start_index = 0
chunk_number = 1

print(f"Starting chunking process...")

while start_index < total_words:
    end_index = min(start_index + chunk_size, total_words)
    chunk_text = " ".join(words[start_index:end_index])
    chunks.append({"text": chunk_text, "chunk_number": chunk_number})
    
    # print(f"  Created chunk {chunk_number}: words {start_index} to {end_index-1}") # Uncomment for detailed log
    
    # Calculate the start of the next chunk
    next_start_index = start_index + chunk_size - overlap
    
    # Ensure progress is made
    if next_start_index <= start_index:
        if end_index == total_words:
             break # Already processed the last part
        next_start_index = start_index + 1 
         
    start_index = next_start_index
    chunk_number += 1
    
    # Safety break (optional)
    if chunk_number > total_words: # Simple safety
        print("Warning: Chunking loop exceeded total word count, breaking.")
        break

print(f"\nText successfully split into {len(chunks)} chunks.")
```

    Starting chunking process...
    
    Text successfully split into 3 chunks.
    

Let's visualize the created chunks using Pandas DataFrame.


```python
print("--- Chunk Details ---")
if chunks:
    # Create a DataFrame for better visualization
    chunks_df = pd.DataFrame(chunks)
    chunks_df['word_count'] = chunks_df['text'].apply(lambda x: len(x.split()))
    display(chunks_df[['chunk_number', 'word_count', 'text']])
else:
    print("No chunks were created (text might be shorter than chunk size).")
print("-" * 25)
```

    --- Chunk Details ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_number</th>
      <th>word_count</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>150</td>
      <td>Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist. She conducted groundbreaking research on radioactivi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>150</td>
      <td>field hospitals. Marie Curie died in 1934 from aplastic anemia, likely caused by her long-term exposure to radiation. Marie was born on November 7...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>84</td>
      <td>with no proper ventilation, as they processed tons of pitchblende ore to isolate radium. Marie Curie established the Curie Institute in Paris, whi...</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------
    

## Step 6: Define the LLM Prompt for Extraction

This is a critical step. We need to carefully instruct the LLM to extract SPO triples in a specific JSON format. We'll define a system prompt (role) and a user prompt template (instructions).

**Key Instructions Emphasized:**
*   Extract `Subject-Predicate-Object` triples.
*   Output *only* a valid JSON array of objects.
*   Each object must have `"subject"`, `"predicate"`, `"object"` keys.
*   Predicates should be concise (1-3 words).
*   All output values must be lowercase.
*   Resolve pronouns to specific entity names.
*   No extra text, explanations, or markdown code fences around the JSON.


```python
# --- System Prompt: Sets the context/role for the LLM --- 
extraction_system_prompt = """
You are an AI expert specialized in knowledge graph extraction. 
Your task is to identify and extract factual Subject-Predicate-Object (SPO) triples from the given text.
Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt.
Extract core entities and the most direct relationship.
"""

# --- User Prompt Template: Contains specific instructions and the text --- 
extraction_user_prompt_template = """
Please extract Subject-Predicate-Object (S-P-O) triples from the text below.

**VERY IMPORTANT RULES:**
1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
7.  **Completeness:** Extract all distinct factual relationships mentioned.

**Text to Process:**
```text
{text_chunk}
```

**Required JSON Output Format Example:**
[
  {{ "subject": "marie curie", "predicate": "discovered", "object": "radium" }},
  {{ "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }}
]

**Your JSON Output (MUST start with '[' and end with ']'):**
"""
```

Let's display the prompts we've defined to verify them.


```python
print("--- System Prompt ---")
print(extraction_system_prompt)
print("\n" + "-" * 25 + "\n")

print("--- User Prompt Template (Structure) ---")
# Show structure, replacing the placeholder for clarity
print(extraction_user_prompt_template.replace("{text_chunk}", "[... text chunk goes here ...]"))
print("\n" + "-" * 25 + "\n")

# Show an example of the *actual* prompt that will be sent for the first chunk
print("--- Example Filled User Prompt (for Chunk 1) ---")
if chunks:
    example_filled_prompt = extraction_user_prompt_template.format(text_chunk=chunks[0]['text'])
    # Displaying a limited portion for brevity
    print(example_filled_prompt[:600] + "\n[... rest of chunk text ...]\n" + example_filled_prompt[-200:])
else:
    print("No chunks available to create an example filled prompt.")
print("\n" + "-" * 25)
```

    --- System Prompt ---
    
    You are an AI expert specialized in knowledge graph extraction. 
    Your task is to identify and extract factual Subject-Predicate-Object (SPO) triples from the given text.
    Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt.
    Extract core entities and the most direct relationship.
    
    
    -------------------------
    
    --- User Prompt Template (Structure) ---
    
    Please extract Subject-Predicate-Object (S-P-O) triples from the text below.
    
    **VERY IMPORTANT RULES:**
    1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
    2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
    3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
    4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
    5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
    6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
    7.  **Completeness:** Extract all distinct factual relationships mentioned.
    
    **Text to Process:**
    ```text
    [... text chunk goes here ...]
    ```
    
    **Required JSON Output Format Example:**
    [
      {{ "subject": "marie curie", "predicate": "discovered", "object": "radium" }},
      {{ "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }}
    ]
    
    **Your JSON Output (MUST start with '[' and end with ']'):**
    
    
    -------------------------
    
    --- Example Filled User Prompt (for Chunk 1) ---
    
    Please extract Subject-Predicate-Object (S-P-O) triples from the text below.
    
    **VERY IMPORTANT RULES:**
    1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
    2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
    3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
    4.  **Lowercase:** ALL
    [... rest of chunk text ...]
    "predicate": "discovered", "object": "radium" },
      { "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }
    ]
    
    **Your JSON Output (MUST start with '[' and end with ']'):**
    
    
    -------------------------
    

## Step 7: LLM Interaction - Extracting Triples (Chunk by Chunk)

Now we loop through each text chunk, send it to the LLM with our prompts, and attempt to parse the expected JSON output. We will show the process for each chunk.


```python
# Initialize lists to store results and failures
all_extracted_triples = []
failed_chunks = []

print(f"Starting triple extraction from {len(chunks)} chunks using model '{llm_model_name}'...")
# We will process chunks one by one in the following cells.
```

    Starting triple extraction from 3 chunks using model 'deepseek-ai/DeepSeek-V3'...
    

### Processing Chunk 1 (Example - loop structure will handle all)


```python
# --- This cell represents the core logic inside the loop for ONE chunk --- 
# --- In a real run, this logic would be in a loop like the original notebook --- 
# --- We show it step-by-step for the first chunk for clarity --- 

chunk_index = 0 # For demonstration, we process only the first chunk here

if chunk_index < len(chunks):
    chunk_info = chunks[chunk_index]
    chunk_text = chunk_info['text']
    chunk_num = chunk_info['chunk_number']
    
    print(f"\n--- Processing Chunk {chunk_num}/{len(chunks)} --- ")
    
    # 1. Format the User Prompt
    print("1. Formatting User Prompt...")
    user_prompt = extraction_user_prompt_template.format(text_chunk=chunk_text)
    # print(f"   Formatted Prompt (Snippet): {user_prompt[:200]}...{user_prompt[-100:]}") # Optional: View prompt
    
    llm_output = None
    error_message = None
    
    try:
        # 2. Make the API Call
        print("2. Sending request to LLM...")
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": extraction_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            # Request JSON output format - helps models that support it
            response_format={ "type": "json_object" }, 
        )
        print("   LLM response received.")
        
        # 3. Extract Raw Response Content
        print("3. Extracting raw response content...")
        llm_output = response.choices[0].message.content.strip()
        print("--- Raw LLM Output (Chunk {chunk_num}) ---")
        print(llm_output)
        print("-" * 20)

    except Exception as e:
        error_message = str(e)
        print(f"   ERROR during API call: {error_message}")
        failed_chunks.append({'chunk_number': chunk_num, 'error': f'API/Processing Error: {error_message}', 'response': ''})

    # 4. Parse JSON (if API call succeeded)
    parsed_json = None
    parsing_error = None
    if llm_output is not None:
        print("4. Attempting to parse JSON from response...")
        try:
            # Strategy 1: Direct parsing (ideal)
            parsed_data = json.loads(llm_output)

            # Handle if response_format={'type':'json_object'} returns a dict containing the list
            if isinstance(parsed_data, dict):
                print("   Detected dictionary response, attempting to extract list...")
                list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    parsed_json = list_values[0]
                    print("      Successfully extracted list from dictionary.")
                else:
                    raise ValueError("JSON object received, but doesn't contain a single list of triples.")
            elif isinstance(parsed_data, list):
                parsed_json = parsed_data
                print("   Successfully parsed JSON list directly.")
            else:
                 raise ValueError("Parsed JSON is not a list or expected dictionary wrapper.")

        except json.JSONDecodeError as json_err:
            parsing_error = f"JSONDecodeError: {json_err}. Trying regex fallback..."
            print(f"   {parsing_error}")
            # Strategy 2: Regex fallback for arrays potentially wrapped in text/markdown
            match = re.search(r'^\s*(\[.*?\])\s*$', llm_output, re.DOTALL)
            if match:
                json_string_extracted = match.group(1)
                print("      Regex found potential JSON array structure.")
                try:
                    parsed_json = json.loads(json_string_extracted)
                    print("      Successfully parsed JSON from regex extraction.")
                    parsing_error = None # Clear previous error
                except json.JSONDecodeError as nested_err:
                    parsing_error = f"JSONDecodeError after regex: {nested_err}"
                    print(f"      ERROR: Regex content is not valid JSON: {nested_err}")
            else:
                 parsing_error = "JSONDecodeError and Regex fallback failed."
                 print("      ERROR: Regex could not find JSON array structure.")
                 
        except ValueError as val_err:
             parsing_error = f"ValueError: {val_err}" # Catches issues with unexpected structure
             print(f"   ERROR: {parsing_error}")

        # --- Show Parsed Result (or error) ---
        if parsed_json is not None:
            print("--- Parsed JSON Data (Chunk {chunk_num}) ---")
            print(json.dumps(parsed_json, indent=2)) # Pretty print the JSON
            print("-" * 20)
        else:
            print(f"--- JSON Parsing FAILED (Chunk {chunk_num}) --- ")
            print(f"   Final Parsing Error: {parsing_error}")
            print("-" * 20)
            failed_chunks.append({'chunk_number': chunk_num, 'error': f'Parsing Failed: {parsing_error}', 'response': llm_output})

    # 5. Validate and Store Triples (if parsing succeeded)
    if parsed_json is not None:
        print("5. Validating structure and extracting triples...")
        valid_triples_in_chunk = []
        invalid_entries = []
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                    # Basic check: ensure values are strings (can be refined)
                    if all(isinstance(item[k], str) for k in ['subject', 'predicate', 'object']):
                        item['chunk'] = chunk_num # Add source chunk info
                        valid_triples_in_chunk.append(item)
                    else:
                        invalid_entries.append({'item': item, 'reason': 'Non-string value'}) 
                else:
                    invalid_entries.append({'item': item, 'reason': 'Incorrect structure/keys'})
        else:
            print("   ERROR: Parsed data is not a list, cannot extract triples.")
            invalid_entries.append({'item': parsed_json, 'reason': 'Not a list'})
            # Also add to failed chunks if the overall structure was wrong
            if not any(fc['chunk_number'] == chunk_num for fc in failed_chunks):
                 failed_chunks.append({'chunk_number': chunk_num, 'error': 'Parsed data not a list', 'response': llm_output})
        
        # --- Show Validation Results --- 
        print(f"   Found {len(valid_triples_in_chunk)} valid triples in this chunk.")
        if invalid_entries:
             print(f"   Skipped {len(invalid_entries)} invalid entries.")
             # print(f"   Invalid entries details: {invalid_entries}") # Uncomment for debugging
             
        # --- Display Valid Triples from this Chunk --- 
        if valid_triples_in_chunk:
             print(f"--- Valid Triples Extracted (Chunk {chunk_num}) ---")
             display(pd.DataFrame(valid_triples_in_chunk))
             print("-" * 20)
             # Add to the main list
             all_extracted_triples.extend(valid_triples_in_chunk)
        else:
             print(f"--- No valid triples extracted from this chunk. ---")
             print("-" * 20)

    # --- Update Running Total (Visual Feedback) ---
    print(f"--- Running Total Triples Extracted: {len(all_extracted_triples)} --- ")
    print(f"--- Failed Chunks So Far: {len(failed_chunks)} --- ")
        
else:
    print(f"Chunk index {chunk_index} is out of bounds (Total chunks: {len(chunks)}). Skipping.")

print("\nFinished processing this chunk.")
# --- IMPORTANT: In a full run, you would uncomment the loop in the original notebook --- 
# --- and remove the `chunk_index = 0` line to process ALL chunks. --- 
```

    
    --- Processing Chunk 1/3 --- 
    1. Formatting User Prompt...
    2. Sending request to LLM...
       LLM response received.
    3. Extracting raw response content...
    --- Raw LLM Output (Chunk {chunk_num}) ---
    [
      { "subject": "marie curie", "predicate": "born as", "object": "maria skłodowska" },
      { "subject": "marie curie", "predicate": "born in", "object": "warsaw, poland" },
      { "subject": "marie curie", "predicate": "was", "object": "physicist" },
      { "subject": "marie curie", "predicate": "was", "object": "chemist" },
      { "subject": "marie curie", "predicate": "conducted", "object": "research on radioactivity" },
      { "subject": "marie curie", "predicate": "discovered", "object": "polonium" },
      { "subject": "marie curie", "predicate": "discovered", "object": "radium" },
      { "subject": "marie curie", "predicate": "was", "object": "first woman to win nobel prize" },
      { "subject": "marie curie", "predicate": "was", "object": "first person to win nobel prize twice" },
      { "subject": "marie curie", "predicate": "was", "object": "only woman to win nobel prize twice" },
      { "subject": "marie curie", "predicate": "was", "object": "only person to win nobel prize in two scientific fields" },
      { "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" },
      { "subject": "marie curie", "predicate": "won", "object": "nobel prize in chemistry" },
      { "subject": "marie curie", "predicate": "developed", "object": "mobile radiography units" },
      { "subject": "marie curie", "predicate": "died in", "object": "1934" },
      { "subject": "marie curie", "predicate": "died from", "object": "aplastic anemia" },
      { "subject": "marie curie", "predicate": "born on", "object": "november 7, 1867" },
      { "subject": "marie curie", "predicate": "born to", "object": "family of teachers" }
    ]
    --------------------
    4. Attempting to parse JSON from response...
       Successfully parsed JSON list directly.
    --- Parsed JSON Data (Chunk {chunk_num}) ---
    [
      {
        "subject": "marie curie",
        "predicate": "born as",
        "object": "maria sk\u0142odowska"
      },
      {
        "subject": "marie curie",
        "predicate": "born in",
        "object": "warsaw, poland"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "physicist"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "chemist"
      },
      {
        "subject": "marie curie",
        "predicate": "conducted",
        "object": "research on radioactivity"
      },
      {
        "subject": "marie curie",
        "predicate": "discovered",
        "object": "polonium"
      },
      {
        "subject": "marie curie",
        "predicate": "discovered",
        "object": "radium"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "first woman to win nobel prize"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "first person to win nobel prize twice"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "only woman to win nobel prize twice"
      },
      {
        "subject": "marie curie",
        "predicate": "was",
        "object": "only person to win nobel prize in two scientific fields"
      },
      {
        "subject": "marie curie",
        "predicate": "won",
        "object": "nobel prize in physics"
      },
      {
        "subject": "marie curie",
        "predicate": "won",
        "object": "nobel prize in chemistry"
      },
      {
        "subject": "marie curie",
        "predicate": "developed",
        "object": "mobile radiography units"
      },
      {
        "subject": "marie curie",
        "predicate": "died in",
        "object": "1934"
      },
      {
        "subject": "marie curie",
        "predicate": "died from",
        "object": "aplastic anemia"
      },
      {
        "subject": "marie curie",
        "predicate": "born on",
        "object": "november 7, 1867"
      },
      {
        "subject": "marie curie",
        "predicate": "born to",
        "object": "family of teachers"
      }
    ]
    --------------------
    5. Validating structure and extracting triples...
       Found 18 valid triples in this chunk.
    --- Valid Triples Extracted (Chunk 1) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>predicate</th>
      <th>object</th>
      <th>chunk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marie curie</td>
      <td>born as</td>
      <td>maria skłodowska</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>marie curie</td>
      <td>born in</td>
      <td>warsaw, poland</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marie curie</td>
      <td>was</td>
      <td>physicist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>marie curie</td>
      <td>was</td>
      <td>chemist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>marie curie</td>
      <td>conducted</td>
      <td>research on radioactivity</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>polonium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>radium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first woman to win nobel prize</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first person to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only woman to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only person to win nobel prize in two scientific fields</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in physics</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in chemistry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>marie curie</td>
      <td>developed</td>
      <td>mobile radiography units</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>marie curie</td>
      <td>died in</td>
      <td>1934</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>marie curie</td>
      <td>died from</td>
      <td>aplastic anemia</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>marie curie</td>
      <td>born on</td>
      <td>november 7, 1867</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>marie curie</td>
      <td>born to</td>
      <td>family of teachers</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    --------------------
    --- Running Total Triples Extracted: 18 --- 
    --- Failed Chunks So Far: 0 --- 
    
    Finished processing this chunk.
    

### Extraction Summary (After Processing All Chunks)

**(Note:** The previous cell only processed *one* chunk for demonstration. In a full run, the loop would process all chunks. The summary below reflects the state *after* the demonstrated single chunk processing. Run the full loop from the original notebook to get the complete results).**

Let's summarize the extraction results and display all accumulated triples.


```python
# --- Summary of Extraction (Reflecting state after the single chunk demo) ---
print(f"\n--- Overall Extraction Summary ---")
print(f"Total chunks defined: {len(chunks)}")
processed_chunks = len(chunks) - len(failed_chunks) # Approximation if loop isn't run fully
print(f"Chunks processed (attempted): {processed_chunks + len(failed_chunks)}") # Chunks we looped through
print(f"Total valid triples extracted across all processed chunks: {len(all_extracted_triples)}")
print(f"Number of chunks that failed API call or parsing: {len(failed_chunks)}")

if failed_chunks:
    print("\nDetails of Failed Chunks:")
    for failure in failed_chunks:
        print(f"  Chunk {failure['chunk_number']}: Error: {failure['error']}")
        # print(f"    Response (start): {failure.get('response', '')[:100]}...") # Uncomment for more detail
print("-" * 25)

# Display all extracted triples using Pandas
print("\n--- All Extracted Triples (Before Normalization) ---")
if all_extracted_triples:
    all_triples_df = pd.DataFrame(all_extracted_triples)
    display(all_triples_df)
else:
    print("No triples were successfully extracted.")
print("-" * 25)
```

    
    --- Overall Extraction Summary ---
    Total chunks defined: 3
    Chunks processed (attempted): 3
    Total valid triples extracted across all processed chunks: 18
    Number of chunks that failed API call or parsing: 0
    -------------------------
    
    --- All Extracted Triples (Before Normalization) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>predicate</th>
      <th>object</th>
      <th>chunk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marie curie</td>
      <td>born as</td>
      <td>maria skłodowska</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>marie curie</td>
      <td>born in</td>
      <td>warsaw, poland</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marie curie</td>
      <td>was</td>
      <td>physicist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>marie curie</td>
      <td>was</td>
      <td>chemist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>marie curie</td>
      <td>conducted</td>
      <td>research on radioactivity</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>polonium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>radium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first woman to win nobel prize</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first person to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only woman to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only person to win nobel prize in two scientific fields</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in physics</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in chemistry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>marie curie</td>
      <td>developed</td>
      <td>mobile radiography units</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>marie curie</td>
      <td>died in</td>
      <td>1934</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>marie curie</td>
      <td>died from</td>
      <td>aplastic anemia</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>marie curie</td>
      <td>born on</td>
      <td>november 7, 1867</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>marie curie</td>
      <td>born to</td>
      <td>family of teachers</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------
    

## Step 8: Normalize and De-duplicate Triples

Now, we clean up the extracted triples:
1.  **Normalize:** Trim whitespace, convert to lowercase.
2.  **Filter:** Remove triples with empty parts after normalization.
3.  **De-duplicate:** Remove exact duplicate `(subject, predicate, object)` combinations.


```python
# Initialize lists and tracking variables
normalized_triples = []
seen_triples = set() # Tracks (subject, predicate, object) tuples
original_count = len(all_extracted_triples)
empty_removed_count = 0
duplicates_removed_count = 0

print(f"Starting normalization and de-duplication of {original_count} triples...")
```

    Starting normalization and de-duplication of 18 triples...
    

We'll iterate through the extracted triples, clean them, and check for duplicates. We'll show the first few transformations.


```python
print("Processing triples for normalization (showing first 5 examples):")
example_limit = 5
processed_count = 0

for i, triple in enumerate(all_extracted_triples):
    show_example = (i < example_limit)
    if show_example:
        print(f"\n--- Example {i+1} ---")
        print(f"Original Triple (Chunk {triple.get('chunk', '?')}): {triple}")
        
    subject_raw = triple.get('subject')
    predicate_raw = triple.get('predicate')
    object_raw = triple.get('object')
    chunk_num = triple.get('chunk', 'unknown')
    
    triple_valid = False
    normalized_sub, normalized_pred, normalized_obj = None, None, None

    if isinstance(subject_raw, str) and isinstance(predicate_raw, str) and isinstance(object_raw, str):
        # 1. Normalize
        normalized_sub = subject_raw.strip().lower()
        normalized_pred = re.sub(r'\s+', ' ', predicate_raw.strip().lower()).strip()
        normalized_obj = object_raw.strip().lower()
        if show_example:
            print(f"Normalized: SUB='{normalized_sub}', PRED='{normalized_pred}', OBJ='{normalized_obj}'")

        # 2. Filter Empty
        if normalized_sub and normalized_pred and normalized_obj:
            triple_identifier = (normalized_sub, normalized_pred, normalized_obj)
            
            # 3. De-duplicate
            if triple_identifier not in seen_triples:
                normalized_triples.append({
                    'subject': normalized_sub,
                    'predicate': normalized_pred,
                    'object': normalized_obj,
                    'source_chunk': chunk_num
                })
                seen_triples.add(triple_identifier)
                triple_valid = True
                if show_example:
                    print("Status: Kept (New Unique Triple)")
            else:
                duplicates_removed_count += 1
                if show_example:
                    print("Status: Discarded (Duplicate)")
        else:
            empty_removed_count += 1
            if show_example:
                print("Status: Discarded (Empty component after normalization)")
    else:
        empty_removed_count += 1 # Count non-string/missing as needing removal
        if show_example:
             print("Status: Discarded (Non-string or missing component)")
    processed_count += 1

print(f"\n... Finished processing {processed_count} triples.")
```

    Processing triples for normalization (showing first 5 examples):
    
    --- Example 1 ---
    Original Triple (Chunk 1): {'subject': 'marie curie', 'predicate': 'born as', 'object': 'maria skłodowska', 'chunk': 1}
    Normalized: SUB='marie curie', PRED='born as', OBJ='maria skłodowska'
    Status: Kept (New Unique Triple)
    
    --- Example 2 ---
    Original Triple (Chunk 1): {'subject': 'marie curie', 'predicate': 'born in', 'object': 'warsaw, poland', 'chunk': 1}
    Normalized: SUB='marie curie', PRED='born in', OBJ='warsaw, poland'
    Status: Kept (New Unique Triple)
    
    --- Example 3 ---
    Original Triple (Chunk 1): {'subject': 'marie curie', 'predicate': 'was', 'object': 'physicist', 'chunk': 1}
    Normalized: SUB='marie curie', PRED='was', OBJ='physicist'
    Status: Kept (New Unique Triple)
    
    --- Example 4 ---
    Original Triple (Chunk 1): {'subject': 'marie curie', 'predicate': 'was', 'object': 'chemist', 'chunk': 1}
    Normalized: SUB='marie curie', PRED='was', OBJ='chemist'
    Status: Kept (New Unique Triple)
    
    --- Example 5 ---
    Original Triple (Chunk 1): {'subject': 'marie curie', 'predicate': 'conducted', 'object': 'research on radioactivity', 'chunk': 1}
    Normalized: SUB='marie curie', PRED='conducted', OBJ='research on radioactivity'
    Status: Kept (New Unique Triple)
    
    ... Finished processing 18 triples.
    

Let's summarize the normalization results and display the final list of unique, clean triples.


```python
# --- Summary of Normalization --- 
print(f"\n--- Normalization & De-duplication Summary ---")
print(f"Original extracted triple count: {original_count}")
print(f"Triples removed (empty/invalid components): {empty_removed_count}")
print(f"Duplicate triples removed: {duplicates_removed_count}")
final_count = len(normalized_triples)
print(f"Final unique, normalized triple count: {final_count}")
print("-" * 25)

# Display a sample of normalized triples using Pandas
print("\n--- Final Normalized Triples ---")
if normalized_triples:
    normalized_df = pd.DataFrame(normalized_triples)
    display(normalized_df)
else:
    print("No valid triples remain after normalization.")
print("-" * 25)
```

    
    --- Normalization & De-duplication Summary ---
    Original extracted triple count: 18
    Triples removed (empty/invalid components): 0
    Duplicate triples removed: 0
    Final unique, normalized triple count: 18
    -------------------------
    
    --- Final Normalized Triples ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>predicate</th>
      <th>object</th>
      <th>source_chunk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marie curie</td>
      <td>born as</td>
      <td>maria skłodowska</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>marie curie</td>
      <td>born in</td>
      <td>warsaw, poland</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marie curie</td>
      <td>was</td>
      <td>physicist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>marie curie</td>
      <td>was</td>
      <td>chemist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>marie curie</td>
      <td>conducted</td>
      <td>research on radioactivity</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>polonium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>marie curie</td>
      <td>discovered</td>
      <td>radium</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first woman to win nobel prize</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>marie curie</td>
      <td>was</td>
      <td>first person to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only woman to win nobel prize twice</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>marie curie</td>
      <td>was</td>
      <td>only person to win nobel prize in two scientific fields</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in physics</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>marie curie</td>
      <td>won</td>
      <td>nobel prize in chemistry</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>marie curie</td>
      <td>developed</td>
      <td>mobile radiography units</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>marie curie</td>
      <td>died in</td>
      <td>1934</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>marie curie</td>
      <td>died from</td>
      <td>aplastic anemia</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>marie curie</td>
      <td>born on</td>
      <td>november 7, 1867</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>marie curie</td>
      <td>born to</td>
      <td>family of teachers</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------
    

## Step 9: Build the Knowledge Graph with NetworkX

Using the clean `normalized_triples`, we construct a `networkx` directed graph (`DiGraph`).
*   Subjects and Objects become nodes.
*   Predicates become edge labels.


```python
# Create an empty directed graph
knowledge_graph = nx.DiGraph()

print("Initialized an empty NetworkX DiGraph.")
# Visualize the initial empty graph state
print("--- Initial Graph Info ---")
try:
    # Try the newer method first
    print(nx.info(knowledge_graph))
except AttributeError:
    # Fallback for different NetworkX versions
    print(f"Type: {type(knowledge_graph).__name__}")
    print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
    print(f"Number of edges: {knowledge_graph.number_of_edges()}")
print("-" * 25)
```

    Initialized an empty NetworkX DiGraph.
    --- Initial Graph Info ---
    Type: DiGraph
    Number of nodes: 0
    Number of edges: 0
    -------------------------
    

Now, we add the triples to the graph one by one, showing the graph's growth.


```python
print("Adding triples to the NetworkX graph...")

added_edges_count = 0
update_interval = 5 # How often to print graph info update

if not normalized_triples:
    print("Warning: No normalized triples to add to the graph.")
else:
    for i, triple in enumerate(normalized_triples):
        subject_node = triple['subject']
        object_node = triple['object']
        predicate_label = triple['predicate']
        
        # Nodes are added automatically when adding edges, but explicit calls are fine too
        # knowledge_graph.add_node(subject_node) 
        # knowledge_graph.add_node(object_node)
        
        # Add the directed edge with the predicate as a 'label' attribute
        knowledge_graph.add_edge(subject_node, object_node, label=predicate_label)
        added_edges_count += 1
        
        # --- Visualize Graph Growth --- 
        if (i + 1) % update_interval == 0 or (i + 1) == len(normalized_triples):
            print(f"\n--- Graph Info after adding Triple #{i+1} --- ({subject_node} -> {object_node})")
            try:
                # Try the newer method first
                print(nx.info(knowledge_graph))
            except AttributeError:
                # Fallback for different NetworkX versions
                print(f"Type: {type(knowledge_graph).__name__}")
                print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
                print(f"Number of edges: {knowledge_graph.number_of_edges()}")
            # For very large graphs, printing info too often can be slow. Adjust interval.

print(f"\nFinished adding triples. Processed {added_edges_count} edges.")
```

    Adding triples to the NetworkX graph...
    
    --- Graph Info after adding Triple #5 --- (marie curie -> research on radioactivity)
    Type: DiGraph
    Number of nodes: 6
    Number of edges: 5
    
    --- Graph Info after adding Triple #10 --- (marie curie -> only woman to win nobel prize twice)
    Type: DiGraph
    Number of nodes: 11
    Number of edges: 10
    
    --- Graph Info after adding Triple #15 --- (marie curie -> 1934)
    Type: DiGraph
    Number of nodes: 16
    Number of edges: 15
    
    --- Graph Info after adding Triple #18 --- (marie curie -> family of teachers)
    Type: DiGraph
    Number of nodes: 19
    Number of edges: 18
    
    Finished adding triples. Processed 18 edges.
    

Let's look at the final graph statistics and sample nodes/edges.


```python
# --- Final Graph Statistics --- 
num_nodes = knowledge_graph.number_of_nodes()
num_edges = knowledge_graph.number_of_edges()

print(f"\n--- Final NetworkX Graph Summary ---")
print(f"Total unique nodes (entities): {num_nodes}")
print(f"Total unique edges (relationships): {num_edges}")

if num_edges != added_edges_count and isinstance(knowledge_graph, nx.DiGraph):
     print(f"Note: Added {added_edges_count} edges, but graph has {num_edges}. DiGraph overwrites edges with same source/target. Use MultiDiGraph if multiple edges needed.")

if num_nodes > 0:
    try:
       density = nx.density(knowledge_graph)
       print(f"Graph density: {density:.4f}")
       if nx.is_weakly_connected(knowledge_graph):
           print("The graph is weakly connected (all nodes reachable ignoring direction).")
       else:
           num_components = nx.number_weakly_connected_components(knowledge_graph)
           print(f"The graph has {num_components} weakly connected components.")
    except Exception as e:
        print(f"Could not calculate some graph metrics: {e}") # Handle potential errors on empty/small graphs
else:
    print("Graph is empty, cannot calculate metrics.")
print("-" * 25)

# --- Sample Nodes --- 
print("\n--- Sample Nodes (First 10) ---")
if num_nodes > 0:
    nodes_sample = list(knowledge_graph.nodes())[:10]
    display(pd.DataFrame(nodes_sample, columns=['Node Sample']))
else:
    print("Graph has no nodes.")

# --- Sample Edges --- 
print("\n--- Sample Edges (First 10 with Labels) ---")
if num_edges > 0:
    edges_sample = []
    for u, v, data in list(knowledge_graph.edges(data=True))[:10]:
        edges_sample.append({'Source': u, 'Target': v, 'Label': data.get('label', 'N/A')})
    display(pd.DataFrame(edges_sample))
else:
    print("Graph has no edges.")
print("-" * 25)
```

    
    --- Final NetworkX Graph Summary ---
    Total unique nodes (entities): 19
    Total unique edges (relationships): 18
    Graph density: 0.0526
    The graph is weakly connected (all nodes reachable ignoring direction).
    -------------------------
    
    --- Sample Nodes (First 10) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Node Sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marie curie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>maria skłodowska</td>
    </tr>
    <tr>
      <th>2</th>
      <td>warsaw, poland</td>
    </tr>
    <tr>
      <th>3</th>
      <td>physicist</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chemist</td>
    </tr>
    <tr>
      <th>5</th>
      <td>research on radioactivity</td>
    </tr>
    <tr>
      <th>6</th>
      <td>polonium</td>
    </tr>
    <tr>
      <th>7</th>
      <td>radium</td>
    </tr>
    <tr>
      <th>8</th>
      <td>first woman to win nobel prize</td>
    </tr>
    <tr>
      <th>9</th>
      <td>first person to win nobel prize twice</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Sample Edges (First 10 with Labels) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marie curie</td>
      <td>maria skłodowska</td>
      <td>born as</td>
    </tr>
    <tr>
      <th>1</th>
      <td>marie curie</td>
      <td>warsaw, poland</td>
      <td>born in</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marie curie</td>
      <td>physicist</td>
      <td>was</td>
    </tr>
    <tr>
      <th>3</th>
      <td>marie curie</td>
      <td>chemist</td>
      <td>was</td>
    </tr>
    <tr>
      <th>4</th>
      <td>marie curie</td>
      <td>research on radioactivity</td>
      <td>conducted</td>
    </tr>
    <tr>
      <th>5</th>
      <td>marie curie</td>
      <td>polonium</td>
      <td>discovered</td>
    </tr>
    <tr>
      <th>6</th>
      <td>marie curie</td>
      <td>radium</td>
      <td>discovered</td>
    </tr>
    <tr>
      <th>7</th>
      <td>marie curie</td>
      <td>first woman to win nobel prize</td>
      <td>was</td>
    </tr>
    <tr>
      <th>8</th>
      <td>marie curie</td>
      <td>first person to win nobel prize twice</td>
      <td>was</td>
    </tr>
    <tr>
      <th>9</th>
      <td>marie curie</td>
      <td>only woman to win nobel prize twice</td>
      <td>was</td>
    </tr>
  </tbody>
</table>
</div>


    -------------------------
    

## Step 10: Visualize the Graph Interactively with ipycytoscape

Finally, we visualize the constructed graph interactively within the notebook using `ipycytoscape`. We'll convert the `networkx` data, define styles, and display the widget.


```python
print("Preparing interactive visualization...")

# --- Check Graph Validity for Visualization --- 
can_visualize = False
if 'knowledge_graph' not in locals() or not isinstance(knowledge_graph, nx.Graph):
    print("Error: 'knowledge_graph' not found or is not a NetworkX graph.")
elif knowledge_graph.number_of_nodes() == 0:
    print("NetworkX Graph is empty. Cannot visualize.")
else:
    print(f"Graph seems valid for visualization ({knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges).")
    can_visualize = True
```

    Preparing interactive visualization...
    Graph seems valid for visualization (19 nodes, 18 edges).
    

### 10.1 Convert NetworkX Data to Cytoscape Format

`ipycytoscape` requires nodes and edges in a specific JSON-like format (list of dictionaries).


```python
cytoscape_nodes = []
cytoscape_edges = []

if can_visualize:
    print("Converting nodes...")
    # Calculate degrees for node sizing
    node_degrees = dict(knowledge_graph.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    for node_id in knowledge_graph.nodes():
        degree = node_degrees.get(node_id, 0)
        # Simple scaling for node size (adjust logic as needed)
        node_size = 15 + (degree / max_degree) * 50 if max_degree > 0 else 15
        
        cytoscape_nodes.append({
            'data': {
                'id': str(node_id), # ID must be string
                'label': str(node_id).replace(' ', '\n'), # Display label (wrap spaces)
                'degree': degree,
                'size': node_size,
                'tooltip_text': f"Entity: {str(node_id)}\nDegree: {degree}" # Tooltip on hover
            }
        })
    print(f"Converted {len(cytoscape_nodes)} nodes.")
    
    print("Converting edges...")
    edge_count = 0
    for u, v, data in knowledge_graph.edges(data=True):
        edge_id = f"edge_{edge_count}" # Unique edge ID
        predicate_label = data.get('label', '')
        cytoscape_edges.append({
            'data': {
                'id': edge_id,
                'source': str(u),
                'target': str(v),
                'label': predicate_label, # Label on edge
                'tooltip_text': f"Relationship: {predicate_label}" # Tooltip on hover
            }
        })
        edge_count += 1
    print(f"Converted {len(cytoscape_edges)} edges.")
    
    # Combine into the final structure
    cytoscape_graph_data = {'nodes': cytoscape_nodes, 'edges': cytoscape_edges}
    
    # Visualize the converted structure (first few nodes/edges)
    print("\n--- Sample Cytoscape Node Data (First 2) ---")
    print(json.dumps(cytoscape_graph_data['nodes'][:2], indent=2))
    print("\n--- Sample Cytoscape Edge Data (First 2) ---")
    print(json.dumps(cytoscape_graph_data['edges'][:2], indent=2))
    print("-" * 25)
else:
     print("Skipping data conversion as graph is not valid for visualization.")
     cytoscape_graph_data = {'nodes': [], 'edges': []}
```

    Converting nodes...
    Converted 19 nodes.
    Converting edges...
    Converted 18 edges.
    
    --- Sample Cytoscape Node Data (First 2) ---
    [
      {
        "data": {
          "id": "marie curie",
          "label": "marie\ncurie",
          "degree": 18,
          "size": 65.0,
          "tooltip_text": "Entity: marie curie\nDegree: 18"
        }
      },
      {
        "data": {
          "id": "maria sk\u0142odowska",
          "label": "maria\nsk\u0142odowska",
          "degree": 1,
          "size": 17.77777777777778,
          "tooltip_text": "Entity: maria sk\u0142odowska\nDegree: 1"
        }
      }
    ]
    
    --- Sample Cytoscape Edge Data (First 2) ---
    [
      {
        "data": {
          "id": "edge_0",
          "source": "marie curie",
          "target": "maria sk\u0142odowska",
          "label": "born as",
          "tooltip_text": "Relationship: born as"
        }
      },
      {
        "data": {
          "id": "edge_1",
          "source": "marie curie",
          "target": "warsaw, poland",
          "label": "born in",
          "tooltip_text": "Relationship: born in"
        }
      }
    ]
    -------------------------
    

### 10.2 Create and Configure the Cytoscape Widget


```python
if can_visualize:
    print("Creating ipycytoscape widget...")
    cyto_widget = ipycytoscape.CytoscapeWidget()
    print("Widget created.")
    
    print("Loading graph data into widget...")
    cyto_widget.graph.add_graph_from_json(cytoscape_graph_data, directed=True)
    print("Data loaded.")
else:
    print("Skipping widget creation.")
    cyto_widget = None
```

    Creating ipycytoscape widget...
    Widget created.
    Loading graph data into widget...
    Data loaded.
    

### 10.3 Define Visual Style

We use a CSS-like syntax to control the appearance of nodes and edges.


```python
if cyto_widget:
    print("Defining enhanced colorful and interactive visual style...")
    # More vibrant and colorful styling with a modern color scheme
    visual_style = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'width': 'data(size)',
                'height': 'data(size)',
                'background-color': '#3498db',  # Bright blue
                'background-opacity': 0.9,
                'color': '#ffffff',             # White text
                'font-size': '12px',
                'font-weight': 'bold',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-wrap': 'wrap',
                'text-max-width': '100px',
                'text-outline-width': 2,
                'text-outline-color': '#2980b9',  # Matching outline
                'text-outline-opacity': 0.7,
                'border-width': 3,
                'border-color': '#1abc9c',      # Turquoise border
                'border-opacity': 0.9,
                'shape': 'ellipse',
                'transition-property': 'background-color, border-color, border-width, width, height',
                'transition-duration': '0.3s',
                'tooltip-text': 'data(tooltip_text)'
            }
        },
        {
            'selector': 'node:selected',
            'style': {
                'background-color': '#e74c3c',  # Pomegranate red
                'border-width': 4,
                'border-color': '#c0392b',
                'text-outline-color': '#e74c3c',
                'width': 'data(size) * 1.2',    # Enlarge selected nodes
                'height': 'data(size) * 1.2'
            }
        },
        {
            'selector': 'node:hover',
            'style': {
                'background-color': '#9b59b6',  # Purple on hover
                'border-width': 4,
                'border-color': '#8e44ad',
                'cursor': 'pointer',
                'z-index': 999
            }
        },
        {
            'selector': 'edge',
            'style': {
                'label': 'data(label)',
                'width': 2.5,
                'curve-style': 'bezier',
                'line-color': '#2ecc71',         # Green
                'line-opacity': 0.8,
                'target-arrow-color': '#27ae60',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 1.5,
                'font-size': '10px',
                'font-weight': 'normal',
                'color': '#2c3e50',
                'text-background-opacity': 0.9,
                'text-background-color': '#ecf0f1',
                'text-background-shape': 'roundrectangle',
                'text-background-padding': '3px',
                'text-rotation': 'autorotate',
                'edge-text-rotation': 'autorotate',
                'transition-property': 'line-color, width, target-arrow-color',
                'transition-duration': '0.3s',
                'tooltip-text': 'data(tooltip_text)'
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'line-color': '#f39c12',         # Yellow-orange
                'target-arrow-color': '#d35400',
                'width': 4,
                'text-background-color': '#f1c40f',
                'color': '#ffffff',               # White text
                'z-index': 998
            }
        },
        {
            'selector': 'edge:hover',
            'style': {
                'line-color': '#e67e22',         # Orange on hover
                'width': 3.5,
                'cursor': 'pointer',
                'target-arrow-color': '#d35400',
                'z-index': 997
            }
        },
        {
            'selector': '.center-node',
            'style': {
                'background-color': '#16a085',    # Teal
                'background-opacity': 1,
                'border-width': 4,
                'border-color': '#1abc9c',        # Turquoise border
                'border-opacity': 1
            }
        }
    ]
    
    print("Setting enhanced visual style on widget...")
    cyto_widget.set_style(visual_style)
    
    # Apply a better animated layout
    cyto_widget.set_layout(name='cose', 
                          nodeRepulsion=5000, 
                          nodeOverlap=40, 
                          idealEdgeLength=120, 
                          edgeElasticity=200, 
                          nestingFactor=6, 
                          gravity=90, 
                          numIter=2500,
                          animate=True,
                          animationDuration=1000,
                          initialTemp=300,
                          coolingFactor=0.95)
    
    # Add a special class to main nodes (Marie Curie)
    if len(cyto_widget.graph.nodes) > 0:
        main_nodes = [node.data['id'] for node in cyto_widget.graph.nodes 
                     if node.data.get('degree', 0) > 10]
        
        # Create gradient styles for center nodes
        for i, node_id in enumerate(main_nodes):
            # Use vibrant colors for center nodes
            center_style = {
                'selector': f'node[id = "{node_id}"]',
                'style': {
                    'background-color': '#9b59b6',   # Purple
                    'background-opacity': 0.95,
                    'border-width': 4,
                    'border-color': '#8e44ad',      # Darker purple border
                    'border-opacity': 1,
                    'text-outline-width': 3,
                    'text-outline-color': '#8e44ad',
                    'font-size': '14px'
                }
            }
            visual_style.append(center_style)
        
        # Update the style with the new additions
        cyto_widget.set_style(visual_style)
    
    print("Enhanced colorful and interactive style applied successfully.")
else:
    print("Skipping style definition.")
```

    Defining enhanced colorful and interactive visual style...
    Setting enhanced visual style on widget...
    Enhanced colorful and interactive style applied successfully.
    

### 10.4 Set Layout Algorithm

We choose an algorithm to automatically arrange the nodes and edges.


```python
if cyto_widget:
    print("Setting layout algorithm ('cose')...")
    # cose (Compound Spring Embedder) is often good for exploring connections
    cyto_widget.set_layout(name='cose', 
                           animate=True, 
                           # Adjust parameters for better spacing/layout
                           nodeRepulsion=4000, # Increase repulsion 
                           nodeOverlap=40,    # Increase overlap avoidance
                           idealEdgeLength=120, # Slightly longer ideal edges
                           edgeElasticity=150, 
                           nestingFactor=5, 
                           gravity=100,        # Increase gravity slightly
                           numIter=1500,      # More iterations
                           initialTemp=200,
                           coolingFactor=0.95,
                           minTemp=1.0)
    print("Layout set. The graph will arrange itself when displayed.")
else:
     print("Skipping layout setting.")
```

    Setting layout algorithm ('cose')...
    Layout set. The graph will arrange itself when displayed.
    

### 10.5 Display the Interactive Graph

The final step is to render the interactive widget in the notebook output below.


```python
if cyto_widget:
    print("Displaying interactive graph widget below...")
    print("Interact: Zoom (scroll), Pan (drag background), Move Nodes (drag nodes), Hover for details.")
    display(cyto_widget)
else:
    print("No widget to display.")

# Add a clear separator
print("\n" + "-" * 25 + "\nEnd of Visualization Step." + "\n" + "-" * 25)
```

    Displaying interactive graph widget below...
    Interact: Zoom (scroll), Pan (drag background), Move Nodes (drag nodes), Hover for details.
    

    PLEAE SEE NOTEBOOK TO VISUALIZE THE INTERACTIVE GRAPH.


    
    -------------------------
    End of Visualization Step.
    -------------------------
    

## Step 11: Conclusion and Next Steps

We have now walked through a very granular process:
1.  Setup libraries and LLM connection.
2.  Defined and chunked input text, visualizing intermediate steps.
3.  Defined detailed prompts for the LLM.
4.  Iterated through chunks (demonstrated with one), showing raw LLM output, parsed JSON, and extracted triples for each.
5.  Aggregated, normalized, and de-duplicated triples, showing the results.
6.  Built the `networkx` graph step-by-step, showing its growth.
7.  Converted data for `ipycytoscape` and visualized the final interactive knowledge graph directly in the notebook.

This detailed breakdown should make the transformation from unstructured text to a structured, visual knowledge graph much clearer.

**Potential Improvements and Further Exploration:**
*   **Run Full Loop:** Execute the LLM extraction and normalization across *all* chunks for a complete graph.
*   **Advanced Normalization:** Implement entity linking or relationship clustering.
*   **Error Handling:** Add retries for LLM calls, better handling of persistent chunk failures.
*   **Prompt Tuning:** Experiment with different models, prompts, and parameters.
*   **Evaluation:** Assess the quality of extracted triples (Precision/Recall).
*   **Richer Visualization:** Use node types for colors/shapes, add community detection coloring, implement more interactive features using ipycytoscape callbacks.
*   **Graph Analysis:** Apply `networkx` algorithms (centrality, paths, etc.).
*   **Persistence:** Store results in a graph database (Neo4j, etc.).
