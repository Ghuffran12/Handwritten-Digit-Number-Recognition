# Handwritten Number Recognition

A local Visual Studio Code project that recognizes a complete handwritten number
(e.g. `5`, `55`, `548`, `12345`) from an uploaded photograph.

## Important accuracy limitation

No OCR model can guarantee 100% accuracy or 100% confidence on every image.
The included model, `microsoft/trocr-base-handwritten`, is a pretrained general
handwriting OCR model. It is not specifically trained on every style of handwritten
digits or photographs of ruled notebook pages. It may misread unusual handwriting,
touching digits, shadows, blur, or digits crossed by notebook lines.

The app deliberately does not invent a confidence percentage. A generated text
sequence's score is not a calibrated probability that the recognized number is correct.

## Requirements

- Python 3.10 or 3.11 recommended.
- Internet connection for the first model download.
- 4 GB RAM minimum recommended; more is helpful.
- NVIDIA GPU is optional. CPU works but may be slower.

## Setup in VS Code (Windows)

1. Extract the project ZIP.
2. Open the `Handwritten_Number_Recognition` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:

   ```bat
   .venv\Scripts\activate.bat
   ```

6. Upgrade pip and install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Streamlit opens the app in your browser. Upload a JPG or PNG photograph and
   click **Recognize handwritten number**.

## First run

The first run downloads the pretrained TrOCR model from Hugging Face. This can take
a few minutes. Later runs use the locally cached model.

## Notebook-line setting

The "Attempt to remove notebook lines" option is OFF by default. If the handwriting
is on ruled paper, try enabling it. If the digits overlap the lines, keep it OFF
because line removal can erase parts of the digits.

## Project structure

```text
Handwritten_Number_Recognition/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Improving accuracy for a specific use case

For dependable recognition on your own photographs, collect a representative,
labelled dataset of your handwritten numbers (including different writers,
paper types, lighting and camera angles), then fine-tune and evaluate the model
on separate training, validation and test sets. Report measured test accuracy;
do not assume 100%.
