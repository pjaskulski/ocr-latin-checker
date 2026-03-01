# **Latin OCR/HTR Checker**

The application is used for automatic analysis and verification of Latin texts originating from **OCR (Optical Character Recognition)** or **HTR (Handwritten Text Recognition)** processes. It is optimized for historical documents and attempts to take into account specific spelling and abbreviations.

The main task of the tool is to indicate potential reading errors, such as character confusion (e.g., ‘v’ instead of ‘u’), segmentation errors (merged or incorrectly separated words), and typos.

## ---

**How does the application work?**

The application is based on two independent analysis engines, which the user can choose depending on their needs:

### **A. Local engine (Hunspell + CLTK)**

It works based on classic natural language processing (NLP) algorithms:

* **Hunspell dictionary (Spylls):** Checks whether a given word exists in the register of classical and medieval Latin.
* **CLTK lemmatization:** Uses the *Classical Language Toolkit* library to reduce words to their base form (lemma). This means that if the word “amamus” is not in the dictionary, but the system recognizes the lemma “amo,” the error weight is reduced.
* **Heuristics:** The system automatically recognizes likely abbreviations (e.g., “V.” or “D.”) and attempts to combine adjacent, incorrect words to see if their combination forms a correct word.

### **B. LLM (Artificial Intelligence) Engine**

Uses the **gpt-5-mini** model (or another configured model) from OpenAI:

* The model analyzes text for semantic and grammatical context.
* It can recognize errors that are invisible to dictionary algorithms (e.g., the use of an incorrect grammatical case that is itself a correct word).
* It returns detailed explanations for each indication.

## ---

**Key interface features**

The interface is designed for readability and interactivity:

* **Input field:** The area where the user pastes the text to be analyzed (up to 5,000 characters).
* **Progress indicator:** Since the analysis (especially local lemmatization) can take from several dozen seconds to a few minutes, after clicking the “Analyze” button, an animation appears informing you about the program's work.
* **Highlighted text:** The results are presented on a copy of the entered text using colored background markers:
* **Red (High):** Probable character confusion error.
  * **Orange (Medium):** Word not found in the dictionary, lexical errors.
  * **Yellow (Low):** Abbreviations or words recognized through lemmatization.

Hovering the mouse cursor over a highlighted word displays detailed information.

* **“List of indications” table:** A detailed report containing the error category, text fragment, justification, and suggested correction. Clicking on the serial number in the table moves the view to the corresponding place in the highlighted text.
* **“Clear” button:** Resets the form and removes all results from the screen.

## ---

**Instalation**

* pip install -r requirements.txt
* python -c "from cltk.data.fetch import FetchCorpus; FetchCorpus(language='lat').import_corpus('lat_models_cltk')"
* python app.py

## ---

**User manual**

1. **Entering text:** Paste the raw text from OCR/HTR into the main text field.
2. **Selecting the mode:**
* Select **“Local”** if you want dictionary verification without sending data outside.
   * Select **“LLM”** if you want the text to be analyzed by a language model.
3. **Analysis:** Click the **“Analyze”** button. Wait until the processing indicator disappears.
4. **Verification:** Review the highlighted fragments. Hover your mouse over the highlight to see a quick tooltip, or check the details in the table below.
5. **Reset:** To start a new job, click **“Clear”**.

## ---

**Technical notes**

* To avoid the application freezing on very long and distorted words, the system automatically skips generating Hunspell suggestions for words longer than 10 characters.
